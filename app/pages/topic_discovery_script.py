
from top2vec import Top2Vec
from wordcloud import WordCloud
from scipy.special import softmax
import pandas as pd
import spacy
from spacy import displacy
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer

NER = spacy.load("en_core_web_sm")

#generates wordcloud for a given topic and all articles in given topic, with boolean input to indicate
#whether hierarchical reduction has been performed and should be accounted for
def wordcloud_generator(model, topic_num, reduction_status):
    if reduction_status:
        word_score_dict = dict(zip(model.topic_words_reduced[topic_num],
                            softmax(model.topic_word_scores_reduced[topic_num])))
    else:
        word_score_dict = dict(
            zip(model.topic_words[topic_num], softmax(model.topic_word_scores[topic_num])))

    cloud = WordCloud(
        width=1600, height=400, background_color="black").generate_from_frequencies(word_score_dict)
    return cloud

def ngram_frequency_df(corpus, ngram):
    if ngram<=2: #for n<=2, remove stopwords
        vec = CountVectorizer(ngram_range=(ngram, ngram), stop_words='english').fit(corpus)
    else: #for n>2, keep stopwords.
        vec = CountVectorizer(ngram_range=(ngram, ngram)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1])
    return words_freq[-20:]

def ngram_bar_visualiser(model, num_topic, reduction_status, ngram_n): #just add column for reduced labels. get_topic_sizes
    docs = list(model.search_documents_by_topic(topic_num = num_topic, num_docs = model.get_topic_sizes()[0][num_topic], reduced=reduction_status))[0]
    common_words = ngram_frequency_df(docs, ngram_n)
    ngram_df = pd.DataFrame(common_words, columns = [ 'phrases', 'count'])
    fig = px.bar(ngram_df, x='count', y='phrases')
    return fig

def display_html_text(text):
    ner_text = NER(text)
    ner_disp = displacy.render(ner_text,style="ent",jupyter=False)
    return ner_disp

#returns top (int art) articles for topic number = num_topic
def articles_generator(model, topic_num, num_articles, reduction_status):
    txt, doc_score = list(model.search_documents_by_topic(topic_num=topic_num, num_docs=num_articles, reduced=reduction_status))[0:2]
    return txt, doc_score
    


    

