from top2vec import Top2Vec
from wordcloud import WordCloud
from scipy.special import softmax
import pandas as pd
from io import BytesIO
#from pyxlsb import open_workbook
import spacy
from spacy import displacy
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sys
sys.path.append("./scripts")
from preprocess_utils import preprocess_utils as preproc

#NER = spacy.load("en_core_web_sm")

def generate_df_topic_labels(model, df, reduction_status):
    df["topic_number"] = model.get_documents_topics(list(range(0,len(df))), reduced = reduction_status)[0]
    topics_ranked_by_fb = list(df.groupby("topic_number", as_index=False).sum("Facebook Interactions").sort_values(by= "Facebook Interactions", ascending=False)["topic_number"])
    df['ranked_topic_number'] = df["topic_number"].apply(lambda topic_num: topics_ranked_by_fb.index(topic_num))
    topics_ranked_by_fb = dict(zip(topics_ranked_by_fb, range(len(topics_ranked_by_fb)))) #{topic_num: ranked_topic_num for ranked_topic_num, topic_num in enumerate(topics_ranked_by_fb)}
    return df, topics_ranked_by_fb

def wordcloud_generator(df, topic_num, ngram, text_column):
    df = df.query("ranked_topic_number == @topic_num")
    df["clean_text"] = preproc.remove_punctuation_df(df, text_column)
    df["clean_text"] = preproc.lowercase_df(df, 'clean_text')
    df["clean_text"] = preproc.full_lemmatization_df(df, 'clean_text')
    #df["clean_text"] = preproc.tokenization(df, 'clean_text')
    if ngram<=2:
        vectorizer = TfidfVectorizer(ngram_range = (ngram,ngram), stop_words = 'english')
    else:
        vectorizer = TfidfVectorizer(ngram_range = (ngram,ngram))
    vecs = vectorizer.fit_transform(df["clean_text"].to_list())
    feature_names = vectorizer.get_feature_names()
    dense = vecs.todense().tolist()
    df = pd.DataFrame(dense, columns=feature_names)
    df.T.sum(axis=1)
    cloud = WordCloud(background_color="white", max_words=30, width=800, height=400).generate_from_frequencies(df.T.sum(axis=1))
    return cloud

def ngram_frequency_df(corpus, ngram): #not in use
    if ngram<=2: #for n<=2, remove stopwords
        vec = CountVectorizer(ngram_range=(ngram, ngram), stop_words='english').fit(corpus)
    else: #for n>2, keep stopwords.
        vec = CountVectorizer(ngram_range=(ngram, ngram)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1])
    return words_freq[-20:]

def ngram_bar_visualiser(model, num_topic, reduction_status, ngram_n): #not in use
    docs = list(model.search_documents_by_topic(topic_num = num_topic, num_docs = model.get_topic_sizes()[0][num_topic], reduced=reduction_status))[0]
    common_words = ngram_frequency_df(docs, ngram_n)
    ngram_df = pd.DataFrame(common_words, columns = [ 'phrases', 'count'])
    fig = px.bar(ngram_df, x='count', y='phrases')
    return fig
   
def df_to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data
