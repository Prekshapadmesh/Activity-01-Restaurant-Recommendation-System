##Importing Libraries

import pandas as pd
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#Loading the cleaned dataset
zomato=pd.read_csv("C:/Users/preks/OneDrive/Desktop/zomato-dataset/clean_data.csv")
print(zomato.head())

## TEXT PRE-PROCESSING

# 5 examples of these columns before text processing:
zomato[['reviews_list', 'cuisines']].sample(5)

## Lower Casing
zomato["reviews_list"] = zomato["reviews_list"].str.lower()
zomato[['reviews_list', 'cuisines']].sample(5)

## Removal of Puctuations
import string
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_punctuation(text))
zomato[['reviews_list', 'cuisines']].sample(5)

## Removal of Stopwords
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_stopwords(text))

## Removal of URLS
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_urls(text))
zomato[['reviews_list', 'cuisines']].sample(5)

zomato.columns = zomato.columns.str.strip()
restaurant_names = list(zomato['name'].unique())
# RESTAURANT NAMES:
restaurant_names = list(zomato['name'].unique())
restaurant_names

# Extracting the most frequent words  
def get_top_words(column, top_nu_of_words, nu_of_word):
    
    vec = CountVectorizer(ngram_range= nu_of_word, stop_words='english')
    
    bag_of_words = vec.fit_transform(column)
    
    sum_words = bag_of_words.sum(axis=0)
    
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    return words_freq[:top_nu_of_words]
zomato.head()
zomato.shape

# Dropping irrelevant columnsw
zomato.columns
zomato=zomato.drop(['address','rest_type', 'type', 'menu_item', 'votes'],axis=1)

df_percent = zomato.sample(frac=0.5)
df_percent.shape

#converting it to csv
df_percent.to_csv('processed_data.csv', index=False)