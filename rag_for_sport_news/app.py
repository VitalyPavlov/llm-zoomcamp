import os
import re
import requests
import time
import pandas as pd
import datetime
from lxml.html import fromstring
from dateutil.parser import parse
from tqdm import tqdm
import threading

from openai import OpenAI
import streamlit as st
from bs4 import BeautifulSoup as bs
import feedparser as fparser

import src.minsearch as minsearch


API_KEY = os.environ.get("OPENAI_API_KEY")
STOP_URL_LIST = ['youtube','instagram','twitter','youtu','shorts','apple','inst','twit','amazo','t.co']
RSS_URL_PATH = './data/rss_url.csv'
PARSED_RSS_PATH = './data/parsed_rss.csv'
PARSED_ARTICLES_PATH = './data/parsed_articles.csv'
PROMT_LIMIT = 30000

client = OpenAI()
client.api_key = API_KEY


def parse_rss() -> None:
    """
    Parser for RSS feeds. Saves the results into a file.
    """

    rss_url = pd.read_csv(RSS_URL_PATH)
    parsed_rss = pd.DataFrame(columns=['date', 'title', 'url', 'summary'])

    for rss in rss_url.rss:
        text = fparser.parse(rss)
        for entry in text.entries:
            date = entry.get("published", "")
            title = entry.get("title", "")
            url = entry.get("link", "")
            summary = entry.get("summary", "")

            wrong_url = any(str(url).lower().__contains__(k) for k in STOP_URL_LIST)
            if not wrong_url:
                parsed_rss.loc[len(parsed_rss)] = [date, title, url, summary]

    if os.path.isfile(PARSED_ARTICLES_PATH):
        articles = pd.read_csv(PARSED_ARTICLES_PATH)
        old_urls = articles.url.values
        parsed_rss = parsed_rss[~parsed_rss.url.isin(old_urls)].reset_index(drop=True)
        parsed_rss.drop_duplicates(subset=['url'], inplace=True)

    parsed_rss.to_csv(PARSED_RSS_PATH, index=False)
    
def parse_articles() -> None:
    """
    Parser for articles (web pages). Saves the results into a file.
    """

    if not os.path.isfile(PARSED_RSS_PATH):
        return
    
    parsed_rss = pd.read_csv(PARSED_RSS_PATH)

    articles = {
        "date": [],
        "url": [],
        "url2": [],
        "title": [],
        "title_rss": [],
        "summary": [],
        "content": []
    }
    
    for i in tqdm(range(len(parsed_rss))):
        date = parsed_rss.loc[i, 'date']
        title_rss = parsed_rss.loc[i, 'title']
        summary = parsed_rss.loc[i, 'summary']
        url = parsed_rss.loc[i, 'url']

        try:
            html = requests.get(url, timeout=(20,20))
            if(html.status_code!=200):
                continue
            url2 = html.url
            html = html.text
            soup = bs(html, 'html.parser').get_text()
            result = re.sub(r'[\t\r\n]', '', soup)
            tree = fromstring(html)
            title = tree.findtext('.//title')
            articles["date"].append(date)
            articles["url"].append(url)
            articles["url2"].append(url2)
            articles['title'].append(title)
            articles['title_rss'].append(title_rss)
            articles['summary'].append(summary)
            articles['content'].append(result)
        except Exception as e:
            continue
    
    articles = pd.DataFrame(articles)
    if os.path.isfile(PARSED_ARTICLES_PATH):
        articles.to_csv(PARSED_ARTICLES_PATH, mode='a', header=False, index=False)
    else:
        articles.to_csv(PARSED_ARTICLES_PATH, index=False)

def clean_articles() -> None:
    """
    Removes articles older 11 days. Saves the results into a file.
    """
    articles = pd.read_csv(PARSED_ARTICLES_PATH)
    articles['date'] = articles['date'].apply(lambda x: parse(x).date())

    end_date = datetime.date.today() 
    start_date = datetime.date.today() - datetime.timedelta(11)

    articles = articles[(articles.date >= start_date) & (articles.date <= end_date)]
    articles.reset_index(inplace=True, drop=True)
    articles.to_csv(PARSED_ARTICLES_PATH, index=False)

    
def update_news(time_step: int=3600) -> None:
    """
    Main function that runs RSS ans articles parsing every time_step second
    """

    while True:
        parse_rss()
        parse_articles()
        clean_articles()
        time.sleep(time_step)
    
def start_hourly_update_news() -> None:
    if not st.session_state.task_running:
        thread = threading.Thread(target=update_news)
        thread.start()
        st.session_state.task_running = True

def build_index(start:int, end:int) -> minsearch.Index:
    """
    Reads the file with articles and generate search index.

    Args:
        start (int): number of days by the date of publication of the article (the beginning of the observation interval)
        end (int): number of days by the date of publication of the article (the end of the observation interval)
    Returns:
        minsearch.Index: index for searching
    """
    art_df = pd.read_csv(PARSED_ARTICLES_PATH)
    art_df.fillna('NA', inplace=True)
    art_df['date'] = art_df['date'].apply(lambda x: parse(x).date())

    end_date = datetime.date.today() - datetime.timedelta(end)
    start_date = datetime.date.today() - datetime.timedelta(start)
    art_df = art_df[(art_df.date >= start_date) & (art_df.date <= end_date)]
    art_df.reset_index(inplace=True, drop=True)

    documents = []
    for i in range(len(art_df)):
        documents.append({
            'text': art_df.loc[i, 'content'],
            'title': art_df.loc[i, 'title'],
            'summary': art_df.loc[i, 'summary'],
            'type': 'article',
        })

    index = minsearch.Index(
        text_fields=["text", "summary", "title"],
        keyword_fields=["article"]
    )
    index.fit(documents)
    return index

def search(query: str, index: minsearch.Index, num_results: int=5):
    """
    Searchs relevant articles in our database.

    Args:
        query (str): our question
        index (minsearch.Index): index for searching
        num_results (int): number of top k articles 
    Returns:
        str: prompt
    """

    boost = {'text': 3.0, 'title': 1.0}

    results = index.search(
        query=query,
        filter_dict={'type': 'article'},
        boost_dict=boost,
        num_results=num_results,
    )

    return results

def build_prompt(query: str, search_results: list) -> str:
    """
    Args:
        query (str): our question
        search_results (list): parsed articles relevant to the question
    Returns:
        str: prompt
    """
    
    prompt_template = """
        You are a football analyst who needs precise information on the latest events in sports.
        Answer the QUESTION and provide a detailed summary based on the CONTEXT from the sport articles, breaking it down into clear, concise points. 
        For each point, give specific details or examples where necessary. Present the information line by line, as a series of highlights with depth.
        Use only the facts from the CONTEXT when answering the QUESTION.

        QUESTION: {question}

        CONTEXT: {context}
    """.strip()

    context = ""
    
    for doc in search_results:
        context = context + f"""
        Title: {doc['title']}
        Text: {doc['text']}\n\n
    """
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt[:PROMT_LIMIT]

def build_cascade_prompt(query: str, search_results: list) -> str:
    """
    Args:
        query (str): our question
        search_results (list): parsed articles relevant to the question
    Returns:
        str: prompt
    """

    prompt_template = """
        You are a football analyst who needs precise information on the latest events in sports.
        Answer the QUESTION based on the CONTEXT from a short review of sport articles, breaking it down into clear, concise points. 
        For each point, give specific details or examples where necessary. Present the information line by line, as a series of highlights with depth.
        Use only the facts from the CONTEXT when answering the QUESTION.

        QUESTION: {question}

        CONTEXT: {context}
    """.strip()

    context = ""
    
    for doc in search_results:
        context = context + f"""{doc}\n\n"""
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm(prompt: str) -> str:
    """
    Run LLM ChatGPT model.

    Args:
        prompt (str): our prompt
    Returns:
        str: LLM response
    """

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def rag(query: str, index: minsearch.Index) -> str:
    """
    The main function that runs RAG components.

    Args:
        query (str): our question
        index (minsearch.Index): index of search
    Returns:
        str: RAG LLM response
    """

    search_results = search(query, index)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer

def cascade_rag(query, index, max_num_results=25, step=5):
    """
    The main function that run RAG components for cascade prediction.
    
    Args:
        query (str): our question
        index (minsearch.Index): index of search
        max_num_results (int): number of all articles to use
        step (int): number of articles in one batch
    Returns:
        str: RAG LLM response
    """

    search_results = search(query, index, num_results=max_num_results)
    cascade_answers = []
    for i in range(0,max_num_results,step):
        prompt = build_prompt(query, search_results[i:i+step])
        answer = llm(prompt)
        cascade_answers.append(answer)
    
    prompt = build_cascade_prompt(query, cascade_answers)
    answer = llm(prompt)
    return answer

def main():
    st.title("RAG for sport news")
    
    if 'slider_values' not in st.session_state:
        st.session_state.slider_values = (1, 3)
    if 'use_cascade' not in st.session_state:
        st.session_state.use_cascade = False
    if 'task_running' not in st.session_state:
        st.session_state.task_running = False
    
    end, start = st.session_state.slider_values
    index = build_index(start, end)

    col1, col2, col3 = st.columns([1,2,1]) 

    with col1:
        if st.button("Update news"):
            start_hourly_update_news()

    with col2:
        st.session_state.slider_values = st.slider("Time interval [days]", 
                                                   1, 10, 
                                                   st.session_state.slider_values, 
                                                   step=1)

    with col3:
        if st.button("Rebuild database"):
            with st.spinner('Updating...'):
                end, start = st.session_state.slider_values
                index = build_index(start, end)
                st.success("Completed!")
    
    user_input = st.text_input("Enter your question:")

    st.session_state.use_cascade = st.checkbox("Cascade prediction", value=st.session_state.use_cascade)

    if st.button("Ask"):
        with st.spinner('Processing...'):
            if st.session_state.use_cascade:
                output = cascade_rag(user_input, index)
            else:
                output = rag(user_input, index)
            st.write(output)

if __name__ == "__main__":
    main()
