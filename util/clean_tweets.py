import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from spellchecker import SpellChecker
import time
stop_words = set(stopwords.words('english'))

# Instantiate lemmatizer
lemmatizer = WordNetLemmatizer()
spell = SpellChecker()

# Top level tweet cleaner
def clean_tweet(tweet):
    # Remove unwanted features like names and hashtags and tokenize
    tokens = remove_unwanted(tweet)
    
    # Tokenize
    # tokens = nltk.word_tokenize(tweet)
    
    # Remove stop words
    tokens = remove_stop_words(tokens)
    # Lemmatize
    tweet_cleaned = lemmatize(tokens)
    tweet_cleaned = [item.lower() for item in tweet_cleaned]
    return tweet_cleaned

# Remove urls, punctiation, hashtags, etc
def remove_unwanted(tweet):
    # Split up by words to remove urls, then put back together
    tweet = remove_urls(tweet.split(' '))
    # Remove hastags, names, and RT
    tweet = remove_twitter_stuff(tweet)
    tweet = ' '.join(tweet)
    # Remove punctuation
    exclude = r'"#$%*+-:-;<=>@[\]^_`{|}~'
    space = r'.,!&()/?'
    tweet = ''.join(ch for ch in tweet if ch not in exclude)
    for c in space:
        tweet = tweet.replace(c, ' ')
    # Get all words
    tweet = re.findall(r"[\w']+", tweet)
    return tweet


def correct_spelling(tokens):
    return [spell.correction(w) for w in tokens] 

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

# tweet = "downloading apps for my iphone! So much fun :-) There literally is an app for just about anything."
# print(clean_tweet(tweet))