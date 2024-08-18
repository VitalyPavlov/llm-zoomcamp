import streamlit as st
import time
import json
import pandas as pd
import os

from elasticsearch import Elasticsearch
from openai import OpenAI
from tqdm import tqdm
import minsearch

client = OpenAI()

tweets_type = "2024 Olympic Men's Basketball articles"


def search(query, index):
    boost = {'text': 3.0, 'title': 1.0}

    results = index.search(
        query=query,
        filter_dict={'tweets_type': tweets_type},
        boost_dict=boost,
        num_results=2
    )

    return results

def build_prompt(query, search_results):
    prompt_template = """
        You are a bookmaker who wants to predict the score of a match fairly accurately.
        Answer the QUESTION based on the CONTEXT from the sport articles.
        Use only the facts from the CONTEXT when answering the QUESTION.

        QUESTION: {question}

        CONTEXT: 
        {context}
    """.strip()

    context = ""
    
    for doc in search_results:
        context = context + f"""
        Title: {doc['title']}
        Text: {doc['text']}\n\n
    """
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm(prompt):
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def rag(query, index):
    search_results = search(query, index)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer

def build_index(tweets_type):
    art_df = pd.read_csv("articles.csv")
    art_df.fillna('NA', inplace=True)

    documents = []
    for i in range(len(art_df)):
        documents.append({
            'text': art_df.loc[i, 'Content'],
            'title': art_df.loc[i, 'Title'],
            'tweets_type': tweets_type,
        })

    index = minsearch.Index(
        text_fields=["text", "title"],
        keyword_fields=["tweets_type"]
    )
    index.fit(documents)
    return index


def main():
    index = build_index(tweets_type)

    st.title("RAG Function Invocation")

    user_input = st.text_input("Enter your input:")

    if st.button("Ask"):
        with st.spinner('Processing...'):
            output = rag(user_input, index)
            st.success("Completed!")
            st.write(output)

if __name__ == "__main__":
    main()
