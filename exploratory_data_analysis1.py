# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk 
from nltk.corpus import stopwords
import string
import re
import emoji
from nltk.stem.snowball import SnowballStemmer
pd.set_option('display.max_colwidth', 100)

# Load dataset
def load_data():
    data = pd.read_csv('output_got.csv')
    return data

def remove_emoji(text):
    """Converts emojis to words
    """
    #text  = "".join([char for char in text if char in emoji.UNICODE_EMOJI])
    text = emoji.demojize(text)
    return text

def tokenization(text):
    text = re.split('\W+', text)
    return text

def remove_stopwords(text):
    """Removes stopwords
    Makes sure to tokenize / put all words in list
    """
    stop_words = set(stopwords.words("english"))
    text = tokenization(text)
    filtered_tweet = [w for w in text if not w in stop_words] 
    return filtered_tweet

def stemming(text):
    """Assumes that text has already been tokenized
    """
    sb = SnowballStemmer("english")
    for word in text:
        print(word, " : ", sb.stem(word)) 

def remove_url(text):
    pattern = r"http\S+"
    #text = "https://www.google.com"
    text = re.sub(pattern, "",text)
    pattern2 = r"www\S+"
    return re.sub(pattern2, "",text)

def remove_hashtag(text):
    pattern = r"#\S+"
    return re.sub(pattern, "",text)

def remove_username(text):
    pattern = r"@\S+"
    return re.sub(pattern, "",text)

#tweet_df = load_data()
#df  = pd.DataFrame(tweet_df[['username', 'text']])
#df['text_noEmoji'] = df['text'].apply(lambda x: remove_emoji(x))
#df['text_nostop'] = df['text_noEmoji'].apply(lambda x: remove_stopwords(x))
#df.head(10)
#df.to_csv("output_got.csv")

#tweet_df.head()

tweet="Testing🤠 Sarah's @what cradle loves pizza and cats but she doesn't herself. #lol https://google.com www.cya.com"
tweet = remove_emoji(tweet)
tweet = remove_url(tweet)
tweet = remove_hashtag(tweet)
tweet = remove_username(tweet)
print(tweet)
print(tokenization(tweet))
tweet = remove_stopwords(tweet)
print(stemming(tweet))
