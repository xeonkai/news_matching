import streamlit as st
import time


from pathlib import Path
import pandas as pd
from top2vec import Top2Vec
import matplotlib.pyplot as plt
from scipy.special import softmax
from wordcloud import WordCloud


st.title("Topic Discovery")
st.sidebar.markdown("# Settings")

start_time = time.perf_counter()

# Load Top2Vec embeddings
# User input: number of clusters - to modify granularity: topic_hierarchy_func
# Eventually: a few news clusters in order of engagement (breaking news)


@st.cache()
def load_news_data(data_path):
    df = pd.read_parquet(data_path)[lambda df: df["source"] == "Online News"]
    return df.copy(), df["title"].to_list(), df["content"].to_list()


data_path = Path("data", "processed", "sg_sanctions_on_russia.parquet")
df, titles, content = load_news_data(data_path)

title_or_content = st.sidebar.selectbox(
    "Compare Title or Content",
    (
        "Content",
        "Title",
    ),
)

st.file_uploader("Upload excel file here")  # NOT LINKED YET


@st.cache()
def load_model(checkpoint, title_or_content):
    if checkpoint == "Top2Vec":
        model_path = Path("scripts", f"top2vec_{title_or_content.lower()}")
        if model_path.is_file():
            model = Top2Vec.load(model_path)
    return model


selected_model = st.sidebar.selectbox("Model Type", ["Top2Vec"])
model = load_model(selected_model, title_or_content)

st.header("Top News Topics")


def wordcloud_generator(mdl, num_topic, reduced=False):
    model._validate_topic_num(num_topic, reduced)
    word_score_dict = dict(
        zip(mdl.topic_words[num_topic], softmax(mdl.topic_word_scores[num_topic]))
    )
    cloud = WordCloud(
        width=1600, height=400, background_color="black"
    ).generate_from_frequencies(word_score_dict)
    return cloud


def top_n_wordclouds(n, a):
    for topicnum in range(n):

        wc = wordcloud_generator(model, topicnum)
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.imshow(wc)
        plt.axis("off")
        plt.title("Topic " + str(topicnum + 1), loc="left", fontsize=30, pad=20)
        with st.container():
            st.pyplot(fig)
            txt = list(model.search_documents_by_topic(topic_num=topicnum, num_docs=a))[
                0
            ]
            st.markdown("#### " + "Sample articles from Topic " + str(topicnum + 1))
            for articlenum in range(a):
                with st.expander("Article " + str(articlenum + 1)):
                    st.markdown("##### Summary of Article")
                    st.markdown("SUMMARISED TEXT PLACEHOLDER")
                    st.markdown("##### Full Article:")
                    st.markdown(txt[articlenum])
            st.text(" ")
            st.text(" ")
            st.text(" ")


j = int(
    st.sidebar.slider(
        label="How many articles?", min_value=1, value=3, max_value=30, step=1
    )
)

k = int(
    st.sidebar.slider(
        label="How many topics?",
        min_value=1,
        value=5,
        max_value=30,
        step=1,
    )
)

top_n_wordclouds(k, j)
