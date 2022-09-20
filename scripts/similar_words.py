import requests
from bs4 import BeautifulSoup
import json


def scraping_similar_phrases(phrases_lst, n):
    phrases_lst = phrases_lst.split(', ')
    similar_lst = []
    for phrase in phrases_lst:
        url = "https://relatedwords.org/relatedto/" + phrase
        req=requests.get(url)
        content=req.text
        soup=BeautifulSoup(content,features="lxml")
        json_string = json.loads(soup.find(id="preloadedDataEl").text)
        similar_lst.append(phrase)
        for wordscore_pairs in json_string["terms"][:n]:
            similar_lst.append(wordscore_pairs['word'])
    return similar_lst