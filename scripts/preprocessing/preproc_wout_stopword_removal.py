import pandas as pd
import numpy as np
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


def filtering_news(data):
    return data.query("source == 'Online News'")

def remove_punctuation_text(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def remove_punctuation_df(data, col):
    return data[col].apply(remove_punctuation_text)

def lowercase_df(data, col):
    return data[col].apply(lambda x: x.lower())

def pos_tagger(nltk_tag):
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

def full_lemmatization_text(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    pos_tagged = nltk.pos_tag(word_tokenize(text))
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
    return data[col].apply(full_lemmatization_text)

def tokenization(data, col):
    return data[col].apply(word_tokenize)

def preprocessing(data, col):
    filtered = filtering_news(data)
    filtered['removed_punc'] = remove_punctuation_df(filtered, 'content')
    filtered['lowercased'] = lowercase_df(filtered, 'removed_punc')
    filtered['clean_content'] = full_lemmatization_df(filtered, 'lowercased')
    return filtered.drop(['removed_punc', 'lowercased'], axis = 1)

