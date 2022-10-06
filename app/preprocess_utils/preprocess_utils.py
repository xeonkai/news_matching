import pandas as pd
import numpy as np
import string

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from nltk.corpus import wordnet
from nltk.corpus import stopwords

stop=set(stopwords.words('english'))

def filtering_news(data):
    """
    
    Filters dataframe entries with source 'Online News' using Pandas library
    
    Args:
        1. raw dataframe (pandas.core.frame.DataFrame)

    Returns: 
        1. filtered dataframe (pandas.core.frame.DataFrame)

    """
    data = data.query("source == 'Online News'")
    return data.reset_index(drop = True)


def remove_punctuation_text(text):
    """
    
    Removes all punctuation in a given string using String library
    
    Args: 
        1. String containing unwanted punctuation (str)

    Returns:
        1. String without unwanted punctuation (str)

    """
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree


def remove_punctuation_df(data, col):
    """
    
    Calls remove_punctuation_text function on each row in a particular column in dataframe, to add a new column that 
    removes all punctuation in a given column using Pandas library

    Args: 
        1. Dataframe (pandas.core.frame.DataFrame)
        2. Column in Dataframe with punctuation to be removed (str)

    Returns:
        1. Updated column with all punctuation removed (pandas.core.series.Series)

    """
    return data[col].apply(remove_punctuation_text)


def lowercase_df(data, col):
    """

    Calls lower() function in String class on each row in a particular column in dataframe, to convert all
    text to lowercase in a given column
    
    Args: 
        1. Dataframe (pandas.core.frame.DataFrame)
        2. Column in Dataframe to be converted to lowercase (str)

    Returns:
        1. Updated column with all text converted to lowercase (pandas.core.series.Series)

    """
    return data[col].apply(lambda x: x.lower())

def pos_tagger(nltk_tag):
    """
    
    Maps Part of Speech (POS) tags assigned by NLTK library's pos_tag function to a format that is accepted 
    as input by WordNet Lemmatizer, by modifying POS tag attached to each token

    Args:
        1. POS tag assigned by pos_tag function (str)

    Returns:
        1. Modified POS tag to be input into WordNet Lemmatizer (str) 

    """
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def full_stemming_text(text):
    """

    Stems text such that each word in text is reduced to its root form

    Args:
        1. String containing unstemmed text (str)

    Returns:
        1. Stemmed text (str)

    """
    return " ".join([PorterStemmer().stem(word) for word in word_tokenize(text)])

def full_stemming_df(data, col):
    """
    
    Applies full_stemming_text to a all entries in a particular column 'col' in dataframe, to stem text

    Args:
        1. Dataframe (pandas.core.frame.DataFrame)
        2. Column in dataframe to be stemmed (str)
    
    Returns:
        1. Updated column with all text stemmed (pandas.core.series.Series)

    """
    return data[col].apply(full_stemming_text)

def full_lemmatization_text(text):
    """
    
    Maps Part of Speech (POS) tags assigned by NLTK library's pos_tag function to a format that is accepted 
    as input by WordNet Lemmatizer, by modifying POS tag attached to each token

    Args:
        1. POS tag assigned by pos_tag function (str)

    Returns:
        1. Modified POS tag to be input into WordNet Lemmatizer (str) 

    """
    wordnet_lemmatizer = WordNetLemmatizer()
    # adds part of speech tagging to input text
    pos_tagged = nltk.pos_tag(word_tokenize(text))
    # creates modified version of pos_tagged list where only nouns, adjectives, verbs, and adverbs are considered
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
        # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:       
        # else use the tag to lemmatize the token
            lemmatized_sentence.append(wordnet_lemmatizer.lemmatize(word, tag))
    lemmatized_sentence = " ".join(lemmatized_sentence)
    return lemmatized_sentence

def full_lemmatization_df(data, col):
    """

    Applies full_lemmatization_text to a all entries in a particular column 'col' in dataframe, to lemmatize text

    Args:
        1. Dataframe (pandas.core.frame.DataFrame)
        2. Column in dataframe to be lemmatized (str)
    
    Returns:
        1. Updated column with all text lemmatized (pandas.core.series.Series)
    
    """
    return data[col].apply(full_lemmatization_text)

def tokenization(data, col):
    """

    Applies word_tokenize from NLTK package to all entries in particular column 'col' in dataframe, to split text 
    into tokens

    Args:
        1. Dataframe (pandas.core.frame.DataFrame)
        2. Column in dataframe to be tokenized (str)

    Returns:
        1. Updated column with each column containing list of strings (pandas.core.series.Series)

    """
    return data[col].apply(lambda x: x.split())

def remove_stopwords_text(text):
    """
    
    Removes stopwords in a string based on if a given word in string appears in "stop" dictionary
    
    Args:
        1. Text containing stopwords (str)

    Returns:
        1. Text with stopwords removed (str)
    
    """
    output= " ".join([i for i in word_tokenize(text) if i not in stop])
    return output

def remove_stopwords_df(data, col):
    """

    Applies remove_stopwords to all entries in particular column 'col' in dataframe, to remove stopwords

    Args:
        1. Dataframe (pandas.core.frame.DataFrame)
        2. Column in dataframe to have stopwords removed (str)

    Returns:
        1. Updated column with each column having stopwords removed (pandas.core.series.Series)

    """
    return data[col].apply(remove_stopwords_text)

def remove_na(data, col):
    """

    Filters entries of dataframe to remove rows where column "col" is empty

    Args:
        1. Dataframe (pandas.core.frame.DataFrame)
        2. Column in dataframe to be conditioned on (str)

    Returns:
        1. Updated dataframe (pandas.core.frame.DataFrame)

    """
    return data[data[col].notna()]

def filter_eng(data):
    """

    Filters entries of dataframe to only keep entries where language is English as per language column

    Args:
        1. Dataframe (pandas.core.frame.DataFrame)

    Returns:
        1. Updated dataframe (pandas.core.frame.DataFrame)

    """
    return data.query("language == 'en'")

def preprocessing(data, col):
    """

    Performs preprocessing with filtering, punctuation removal, lowercasing, and lemmatization

    Args:
        1. Dataframe (pandas.core.frame.DataFrame)
        2. Column in dataframe to have preprocessing performed (str)
    
    Returns:
        1. Updated dataframe with an additional column of "clean_content" with completed
        preprocessing (pandas.core.frame.DataFrame)
 
    """
    filtered = filtering_news(data)
    filtered['removed_punc'] = remove_punctuation_df(filtered, col)
    filtered['lowercased'] = lowercase_df(filtered, 'removed_punc')
    filtered[f'clean_{col}'] = full_lemmatization_df(filtered, 'lowercased')
    return filtered.drop(['removed_punc', 'lowercased'], axis = 1)

def tokenised_preprocessing(data, col):
    """

    Performs preprocessing with filtering, punctuation removal, lowercasing, and tokenisation

    Args:
        1. Dataframe (pandas.core.frame.DataFrame)
        2. Column in dataframe to have preprocessing performed (str)
    
    Returns:
        1. Updated dataframe with an additional column of "clean_content" with completed
        preprocessing with tokenisation (pandas.core.frame.DataFrame). Each row of clean_content contains a list of tokens.
 
    """
    #filtered = filtering_news(data)
    data['removed_punc'] = remove_punctuation_df(data, col)
    data['lowercased'] = lowercase_df(data, 'removed_punc')
    data[f'clean_{col}'] = tokenization(data, 'lowercased')
    return data.drop(['removed_punc', 'lowercased'], axis = 1)

def full_preprocessing(data, col):
    """

    Performs full preprocessing with filtering, punctuation removal, lowercasing, lemmatization
    and stopword removal

    Args:
        1. Dataframe (pandas.core.frame.DataFrame)
        2. Column in dataframe to have preprocessing performed (str)
    
    Returns:
        1. Updated dataframe with an additional column of "clean_content" with completed
        preprocessing (pandas.core.frame.DataFrame)
 
    """
    filtered = filtering_news(data)
    filtered['removed_punc'] = remove_punctuation_df(filtered, col)
    filtered['lowercased'] = lowercase_df(filtered, 'removed_punc')
    filtered['lemmatized'] = full_lemmatization_df(filtered, 'lowercased')
    filtered[f'clean_{col}'] = remove_stopwords_df(filtered, 'lemmatized')
    return filtered.drop(['removed_punc', 'lowercased', 'lemmatized'], axis = 1)
