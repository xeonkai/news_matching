import streamlit as st
import time
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
from pathlib import Path
import pandas as pd
from top2vec import Top2Vec
import matplotlib.pyplot as plt
from scipy.special import softmax
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import topic_discovery_script as td

@st.cache(allow_output_mutation=True)
def load_news_data(data_path):
    df = pd.read_parquet(data_path)[lambda df: df["source"] == "Online News"]
    return df.copy(), df["title"].to_list(), df["content"].to_list(), df["date"].to_list()

data_path = Path("data", "processed", "sg_sanctions_on_russia.parquet")
df, titles, content, dates = load_news_data(data_path)


upstream_tab, td_tab = st.tabs(["Dataset Filters", "Topic Discovery"])

with upstream_tab:
    #date_filters = st.sidebar.date_input(
    #            "Article dates",
    #            value=(min(dates), max(dates)),
    #            min_value=min(dates),
    #            max_value=max(dates),
    #    ) #TODO: not implemented

    upstream_keywords = str(
        st.sidebar.text_input(
            label = "Filter to keep articles containing specific keywords", 
            value = ""
        )
    )

    upstream_engagement_limit = int(
        st.sidebar.slider(
            label = "Minimum value of Facebook engagement", 
            min_value = 1,
            max_value = 10
        )
    )

    selected_model = st.sidebar.selectbox("Model Type", ["Top2Vec"])
    

with td_tab:
    st.title("Topic Discovery")
    st.sidebar.markdown("# Settings")

    start_time = time.perf_counter()


    #load news data as parquet file for faster processing
    @st.cache(allow_output_mutation=True)
    def load_news_data(data_path):
        df = pd.read_parquet(data_path)[lambda df: df["source"] == "Online News"]
        return df.copy(), df["title"].to_list(), df["content"].to_list(), df["date"].to_list()


    data_path = Path("data", "processed", "sg_sanctions_on_russia.parquet")
    df, titles, content, dates = load_news_data(data_path)

    #TODO: link titles
    title_or_content = st.sidebar.selectbox(
        "Compare Title or Content",
        (
            "Title",
            "Content"
        ),
    )

    st.file_uploader("Upload excel file here")  #TODO


    @st.cache(allow_output_mutation=True)
    def load_model(checkpoint, title_or_content):
        if checkpoint == "Top2Vec":
            model_path = Path("scripts", f"top2vec_{title_or_content.lower()}")
            if model_path.is_file():
                model = Top2Vec.load(model_path)
        return model

    model = load_model(selected_model, title_or_content)

    st.header("Top News Topics") 


    def top_n_topics(model, num_topics, num_articles, keywords, topic_granularity, ngram_value):
        if topic_granularity != model.get_num_topics():
            model.hierarchical_topic_reduction(num_topics=topic_granularity)
            red_status = True
        else:
            red_status = False
        if keywords != "": #TODO: create an exception if any words in kw have not been learned by model
            tokenized_kw = nltk.word_tokenize(keywords)
            subset_topics = model.search_topics(keywords=tokenized_kw, num_topics=num_topics, reduced=red_status)[3] 
        else:
            subset_topics = model.get_topic_sizes(reduced=red_status)[1][0:num_topics]
        for topic_num in subset_topics:
            #wc = wordcloud_generator(topicnum, red)
            #fig, ax = plt.subplots(figsize=(32, 8))
            #ax.imshow(wc)
            #plt.axis("off")
            #plt.title("Topic " + str(topicnum + 1), loc="left", fontsize=30, pad=20)
            #st.pyplot(fig)
            fig = td.ngram_bar_visualiser(model, topic_num, red_status, ngram_value)
            st.plotly_chart(fig)
            text, doc_score = td.articles_generator(model, topic_num, num_articles, red_status) 
            st.text("Number of articles in topic: " + str(model.get_topic_sizes(reduced=red_status)[0][topic_num]))
            st.markdown("#### " + "Sample articles from Topic " + str(topic_num + 1))
            #iterates through number of articles to be returned
            for article_num in range(num_articles):
                with st.expander("Article " + str(article_num + 1)):
                    st.markdown("##### Full Article:")
                    ent_html = td.display_html_text(text[article_num])
                    st.markdown(ent_html, unsafe_allow_html=True) #full article text
                #st.text(doc_score[articlenum]) #semantic similarity of document to topic (by cosine sim)
            st.text(" ")
            st.text(" ")
            st.text(" ") 

    ngram_size = int(
        st.sidebar.slider(
            label="How many words should be in identified phrases?", min_value=1, value=2, max_value=4, step=1
        )
    )

    num_of_articles = int(
        st.sidebar.slider(
            label="How many articles to be displayed per topic?", min_value=1, value=3, max_value=30, step=1
        )
    )

    num_of_topics = int(
        st.sidebar.slider(
            label="How many topics to be displayed?",
            min_value=1,
            value=5,
            max_value=30,
            step=1,
        )
    )

    topic_granularity = int(
        st.sidebar.slider(
            label="How granular should topics be?", 
            min_value=1, 
            value=model.get_num_topics(), 
            max_value=model.get_num_topics(),
            step=1
        )
    )

    downstream_keywords = str(
        st.sidebar.text_input("Search for keywords in topics", ""
        )
    )

    top_n_topics(model, num_of_topics, num_of_articles, downstream_keywords, topic_granularity, ngram_size)