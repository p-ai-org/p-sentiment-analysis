import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import time

''' Methods for cleaning tweets '''

stop_words = set(stopwords.words('english'))
# Let's keep 'not' in the tweets
stop_words.remove('not')

# Instantiate lemmatizer
lemmatizer = WordNetLemmatizer()
tweet_tokenizer = TweetTokenizer()

# Top level tweet cleaner
def clean_tweet(tweet):
    tweet = str(tweet)
    # Let's just split up words with apostrophes
    tokens = tweet.replace("'", ' ')
    # Tokenize
    tokens = tweet_tokenizer.tokenize(tokens)
    # Remove unwanted features like names and hashtags and tokenize
    tokens = remove_unwanted(tokens)
    # Lowercase
    tokens = [x.lower() for x in tokens]
    # Clean up after splitting by apostrophe
    tokens = [clean_concatenations(x) for x in tokens]
    # Remove stop words
    tokens = remove_stop_words(tokens)
    # Lemmatize
    # tokens = lemmatize(tokens)
    return tokens

# Remove urls, punctiation, hashtags, etc
def remove_unwanted(tokens):
    # Remove URLs
    tweet = remove_urls(tokens)
    # Remove hastags, names, and RT
    tweet = remove_twitter_stuff(tweet)
    # Remove punctuation
    exclude = re.compile(r'^[,\.!?/\-"#$%&\'\(\)\*\+:;\[\]\\^_{|}~]*$')
    tweet = [x for x in tweet if not exclude.search(x)]
    return tweet

# Clean up common results of splitting by apostrophe (also deal with 'u' abbreviation)
def clean_concatenations(token):
    if token == "t":
        return 'not'
    # If it's just one letter, return something that'll get taken out by stop_words
    elif len(token) == 1:
        return 'is'
    return token

def lemmatize(tokens):
    return [lemmatizer.lemmatize(w, get_pos(w)) for w in tokens]

# Get part of speech
def get_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Remove hashtags, names, and RT
def remove_twitter_stuff(tweet_tokens):
    regex = re.compile(r'^#.+|^@.+|RT')
    return [i for i in tweet_tokens if not regex.search(i)]

# Remove stop words
def remove_stop_words(tweet_tokens):
    return [w for w in tweet_tokens if not w in stop_words]

# Remove any words with hhtp or .com
def remove_urls(tweet_tokens):
    regex = re.compile(r'^http|.*\.com.*')
    return [i for i in tweet_tokens if not regex.search(i)]