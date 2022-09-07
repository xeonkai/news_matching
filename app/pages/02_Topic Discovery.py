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
import spacy
from spacy import displacy

NER = spacy.load("en_core_web_sm")

st.title("Topic Discovery")
st.sidebar.markdown("# Settings")

start_time = time.perf_counter()

# streamlit run "/Users/shreya/Documents/govtech/news_matching/app/pages/02_Topic Discovery.py"
# next: add date, fix $ issue, try named entity recognition


#load news data as parquet file for faster processing
@st.cache(allow_output_mutation=True)
def load_news_data(data_path):
    df = pd.read_parquet(data_path)[lambda df: df["source"] == "Online News"]
    return df.copy(), df["title"].to_list(), df["content"].to_list()


data_path = Path("data", "processed", "sg_sanctions_on_russia.parquet")
df, titles, content = load_news_data(data_path)

#TODO: link titles
title_or_content = st.sidebar.selectbox(
    "Compare Title or Content",
    (
        "Content",
        "Title",
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

#loads model (with pre-computed embeddings) based on selected_model selection
selected_model = st.sidebar.selectbox("Model Type", ["Top2Vec"])
model = load_model(selected_model, title_or_content)

st.header("Top News Topics") 

#this function checks if hierarchical reduction has been perfored on Top2Vec model yet. if yes, return True.
def hierarchical_reduction(num_h):
    if num_h==model.get_num_topics():
        return False
    else:
        return True

#generates wordcloud for a given topic and all articles in given topic, with boolean input to indicate
#whether hierarchical reduction has been performed and should be accounted for
def wordcloud_generator(num_topic, red):
    if red:
        model._validate_hierarchical_reduction()
        model._validate_topic_num(num_topic, red)
        word_score_dict = dict(zip(model.topic_words_reduced[num_topic],
                            softmax(model.topic_word_scores_reduced[num_topic])))
    else:
        model._validate_topic_num(num_topic, red)
        word_score_dict = dict(
            zip(model.topic_words[num_topic], softmax(model.topic_word_scores[num_topic])))

    cloud = WordCloud(
        width=1600, height=400, background_color="black").generate_from_frequencies(word_score_dict)
    return cloud

#returns a sorted dataframe of words/phrases of maximum frequency 
def get_top_n_gram(corpus, ngram, n): #add some customisation: for n<=2, remove stopwords. for n>2, keep stopwords.
    if ngram<=2:
        vec = CountVectorizer(ngram_range=(ngram, ngram), stop_words='english').fit(corpus)
    else:
        vec = CountVectorizer(ngram_range=(ngram, ngram)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

#returns an n-gram plot of words/phrases with maximum frequency for topic number = num_topic
#control "n in n-gram" with ngram_n parameter
def ngram_generator(num_topic, red, ngram_n): #just add column for reduced labels. get_topic_sizes
    topic_labels = model.get_documents_topics(list(range(0,len(df))), reduced=red)[0]
    df['topic_num'] = topic_labels
    common_words = get_top_n_gram(df[df['topic_num']==num_topic]['content'], ngram_n, 20)
    ngram_df = pd.DataFrame(common_words, columns = ['phrases' , 'count'])
    fig = px.bar(ngram_df, x='phrases', y='count')
    return fig

def display_html_text(text):
    ner_text = NER(text)
    ner_disp = displacy.render(ner_text,style="ent",jupyter=False)
    return ner_disp

#returns top (int art) articles for topic number = num_topic
def articles_generator(num_topic, art, red):
    # note: for keyword example, doesnt return articles closest to keyword
    # instead returns documents closest to the topic vector that is closest to keywords
    txt, doc_score = list(model.search_documents_by_topic(topic_num=num_topic, num_docs=art, reduced=red))[0:2]
    st.text("Number of articles in topic: " + str(model.get_topic_sizes(reduced=red_status)[0][num_topic]))
    st.markdown("#### " + "Sample articles from Topic " + str(num_topic + 1))
    #iterates through number of articles to be returned
    for articlenum in range(art):
        with st.expander("Article " + str(articlenum + 1)):
            st.markdown("##### Summary of Article")
            st.markdown("SUMMARISED TEXT PLACEHOLDER")
            st.markdown("##### Full Article:")
            ent_html = display_html_text(txt[articlenum])
            st.markdown(ent_html, unsafe_allow_html=True) #full article text
            #st.text(doc_score[articlenum]) #semantic similarity of document to topic (by cosine sim)
    st.text(" ")
    st.text(" ")
    st.text(" ")

#n: number of topics to print
#a: number of articles per topic to print
#kw: string of keywords to look out for in topics (if any)
#num_h: granularity of topics (total number of topic clusters desired from model - hierarchical reduction performed if red=True)
#ngram_val: indicates 'n' value in n-gram
#red: status of whether hierarchical reduction should be performed
def top_n_topics(n, a, kw, num_h, ngram_val, red):
    if red==True:
        model.hierarchical_topic_reduction(num_topics=num_h)
    if kw!="": #TODO: create an exception if any words in kw have not been learned by model
        tokenized_kw = nltk.word_tokenize(kw)
        topic_nums = model.search_topics(keywords=tokenized_kw, num_topics=n, reduced=red)[3] 
    else:
        topic_nums = model.get_topic_sizes(reduced=red)[1][0:n]
    for topicnum in topic_nums:
        #wc = wordcloud_generator(topicnum, red)
        #fig, ax = plt.subplots(figsize=(32, 8))
        #ax.imshow(wc)
        #plt.axis("off")
        #plt.title("Topic " + str(topicnum + 1), loc="left", fontsize=30, pad=20)
        #st.pyplot(fig)
        fig = ngram_generator(topicnum, red, ngram_val)
        fig.update_xaxes(tickangle=35)
        st.plotly_chart(fig)
        articles_generator(topicnum, a, red)  

ng = int(
    st.sidebar.slider(
        label="How many words should be in identified phrases?", min_value=1, value=2, max_value=4, step=1
    )
)

j = int(
    st.sidebar.slider(
        label="How many articles to be displayed per topic?", min_value=1, value=3, max_value=30, step=1
    )
)

k = int(
    st.sidebar.slider(
        label="How many topics to be displayed?",
        min_value=1,
        value=5,
        max_value=30,
        step=1,
    )
)

h = int(
    st.sidebar.slider(
        label="How granular should topics be?", 
        min_value=1, 
        value=model.get_num_topics(), 
        max_value=model.get_num_topics(),
        step=1
        #on_change=hierarchical_reduction()
    )
)

text = str(
    st.sidebar.text_input("Search for keywords in topics", "")
)

red_status=False
red_status = hierarchical_reduction(h) #sets red_status to Boolean value of True of False
top_n_topics(k, j, text, h, ng, red_status)