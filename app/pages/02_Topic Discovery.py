import streamlit as st
import time
import os

#os.chdir("/Users/shreya/Documents/govtech/news_matching")

from pathlib import Path
import pandas as pd
from top2vec import Top2Vec
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from wordcloud import WordCloud

#raw_summ_path = Path("scripts", "summarisation.py")


#from ..scripts.summarisation import summarise_text

#from summarisation import summarise_text

#streamlit run "/Users/shreya/Documents/govtech/news_matching/app/pages/02_Article Discovery.py" 

st.title("Topic Discovery")
st.sidebar.markdown("# Settings")

start_time = time.perf_counter()

# Load Top2Vec embeddings
# User input: number of clusters - to modify granularity: topic_hierarchy_func
# Eventually: a few news clusters in order of engagement (breaking news)

raw_data_path = Path("data", "raw", "SG sanctions on Russia.xlsx")

@st.cache()
def load_news_data():
    #SET TO UKRAINE ARTICLES FOR NOW
    df = (
        pd.read_excel(
            raw_data_path,
            sheet_name="Contents",
            parse_dates=["date"],
            usecols=[
                "id",
                "source",
                "title",
                "content",
                "date",
                "url",
                "domain",
            ]
        ).set_index("id") 
    )[lambda df: df["source"] == "Online News"]
    return df.copy(), df["title"].to_list(), df["content"].to_list()

df, titles, content = load_news_data()

st.sidebar.selectbox(
    "Compare Title or Content",
    (
        "Title",
        "Content",
    ),
)

st.file_uploader('Upload excel file here') #NOT LINKED YET

@st.cache()
def load_model(chosen_model):
    if chosen_model=="Top2Vec":
        if Path("scripts/top2vec").is_file():
            model = Top2Vec.load("scripts/top2vec")
        else:
            model = Top2Vec(documents=content, speed="learn", workers=8) #Top2Vec(documents=list(data["content"]), speed="learn", workers=8)
    return model 

selected_model = st.sidebar.selectbox("Model Type", ['Top2Vec'])
model = load_model(selected_model)

st.header("Top News Topics")

def wordcloud_generator(mdl, num_topic, reduced=False):
    model._validate_topic_num(num_topic, reduced)
    word_score_dict = dict(zip(mdl.topic_words[num_topic], softmax(mdl.topic_word_scores[num_topic])))
    cloud = WordCloud(width=1600,
                      height=400,
                      background_color='black').generate_from_frequencies(word_score_dict)
    return cloud

def top_n_wordclouds(n, a):
    for topicnum in range(n):

        wc = wordcloud_generator(model, topicnum)
        fig, ax = plt.subplots(figsize = (16, 4))
        ax.imshow(wc)
        plt.axis("off")
        plt.title("Topic " + str(topicnum+1), loc='left', fontsize=30, pad=20)
        with st.container():
            st.pyplot(fig)
            txt = list(model.search_documents_by_topic(topic_num=topicnum, num_docs=a))[0]
            st.markdown("#### " + "Sample articles from Topic " + str(topicnum+1))
            for articlenum in range(a):
                with st.expander("Article " + str(articlenum+1)):
                    st.markdown("##### Summary of Article")
                    st.markdown("SUMMARISED TEXT PLACEHOLDER")
                    st.markdown("##### Full Article:")
                    st.markdown(txt[articlenum])
            st.text(" "); st.text(" "); st.text(" ")  


j = int(
    st.sidebar.slider(
        label="How many articles?",
        min_value=1,
        value=3,
        max_value=30,
        step=1
    )
)

k = int(
    st.sidebar.slider(
        label="How many topics?",
        min_value=1,
        value=5,
        max_value = 30,
        step=1,
    )
)

#def discover_top_news_topics():





top_n_wordclouds(k, j)






