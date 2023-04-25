from wordcloud import WordCloud
import pandas as pd
from io import BytesIO
import plotly.express as px
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sys
sys.path.append("./scripts")
from preprocess_utils import preprocess_utils as preproc

def add_full_text(df):
    if "Summary" in df.columns:
        df["full_text"] = df[["Headline", "Summary"]].agg(" ".join, axis = 1)
    else:
        df["full_text"] = df["Headline"]
    return df

# to change this function
def generate_df_topic_labels(model, df, reduction_status):
    """
    Adds an additional column to dataframe of articles to rank topics output by topic modelling, in descending order
    of their sum of Facebook Interactions. Also outputs a list of ranked topic numbers

    Input:

    model - Top2Vec saved model
    df (pandas.DataFrame) - input dataframe with topic labels
    reduction_status (Bool) - Indicator of whether hierarchical reduction has been performed on model

    Output:

    df (pandas.DataFrame) - dataframe with additional column of "ranked topic number"
    topics_ranked_by_fb (list) - topic number sorted in order of descending Facebook engagement

    """
    df["topic_number"] = model.get_documents_topics(list(range(0,len(df))), reduced = reduction_status)[0]
    topics_ranked_by_fb = list(df.groupby("topic_number", as_index=False).sum("Facebook Interactions").sort_values(by= "Facebook Interactions", ascending=False)["topic_number"])
    df['ranked_topic_number'] = df["topic_number"].apply(lambda topic_num: topics_ranked_by_fb.index(topic_num))
    topics_ranked_by_fb = dict(zip(topics_ranked_by_fb, range(len(topics_ranked_by_fb)))) #{topic_num: ranked_topic_num for ranked_topic_num, topic_num in enumerate(topics_ranked_by_fb)}
    
    #added 4/4/23
    #to arrange topics on topic discovery page based on highest no. of fb interactions, then calc next closest cluster based on euclidean distance
    topic_vectors = list(model.topic_vectors) #get topic vectors
    curr_topic = list(topics_ranked_by_fb.keys())[0] #get topic no. with highest number of fb interactions
    final_ranked = {} #key: topic num, value: ranked topic num
    count = 0
    final_ranked[curr_topic] = 0 #set curr_topic to highest ranked topic
    curr_vector = topic_vectors[curr_topic]
    num_topics = len(topics_ranked_by_fb)
    while num_topics > 1:
        #calc distance of other topic vectors to current topic vector
        count += 1
        topic_distance = {i: np.linalg.norm(topic_vectors[i] - curr_vector) for i in range(len(topic_vectors))}
        sorted_topic_distance = dict(sorted(topic_distance.items(), key=lambda x: x[1]))
        curr_topic = list({key: sorted_topic_distance[key] for key in sorted_topic_distance if key not in final_ranked}.keys())[0] #choose next topic with topic vector closest to the current topic
        final_ranked[curr_topic] = count
        num_topics -= 1
    return df, final_ranked

def wordcloud_generator(df, topic_num, ngram, text_column):
    """
    Generates N-Gram phrase cloud based on word/phrase frequency in texts

    Input:

    df (pandas.DataFrame) - input dataframe with topic labels
    topic_num (int) - topic number
    ngram (int) - expected number of words in wordcloud phrases
    text_column (int) - name of column in input dataframe with text 

    Output:

    cloud - n-gram phrase cloud
    
    """
    df = df.query("ranked_topic_number == @topic_num")
    df["clean_text"] = preproc.remove_punctuation_df(df, text_column)
    df["clean_text"] = preproc.lowercase_df(df, 'clean_text')
    df["clean_text"] = preproc.full_lemmatization_df(df, 'clean_text')
    if ngram<=2:
        vectorizer = CountVectorizer(ngram_range = (ngram,ngram), stop_words = 'english')
    else:
        vectorizer = CountVectorizer(ngram_range = (ngram,ngram))
    vecs = vectorizer.fit_transform(df["clean_text"].to_list())
    feature_names = vectorizer.get_feature_names()
    dense = vecs.todense().tolist()
    df = pd.DataFrame(dense, columns=feature_names)
    cloud = WordCloud(background_color="white", max_words=30, width=800, height=400).generate_from_frequencies(df.T.sum(axis=1))
    return cloud

def df_to_excel(df):
    """
    Convert dataframe to Excel file for download in Streamlit

    Input:
    df (pandas.DataFrame) - input dataframe

    Output:
    processed_data - output dataframe for Excel export
    
    """

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

def df_to_excel_both(df1, df2):
    """
    Convert dataframe to Excel file for download in Streamlit

    Input:
    df (pandas.DataFrame) - input dataframe

    Output:
    processed_data - output dataframe for Excel export
    
    """

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df1.to_excel(writer, index=False, sheet_name='Sheet1')
    df2.to_excel(writer, index=True, sheet_name='Sheet2')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data
