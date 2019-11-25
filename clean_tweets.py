import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import time
stop_words = set(stopwords.words('english'))

# Instantiate lemmatizer
lemmatizer = WordNetLemmatizer()

# Top level tweet cleaner
def clean_tweet(tweet):
    # Remove unwanted features like names and hashtags
    tweet = remove_unwanted(tweet)
    # Tokenize
    tokens = nltk.word_tokenize(tweet)
    # Remove stop words
    tokens = remove_stop_words(tokens)
    # Lemmatize
    tweet_cleaned = lemmatize(tokens)
    return tweet_cleaned

# Remove urls, punctiation, hashtags, etc
def remove_unwanted(tweet):
    tweet = remove_urls(tweet.split(' '))
    tweet = ' '.join(tweet)
    # Remove punctuation
    exclude = set(string.punctuation)
    tweet = ''.join(ch for ch in tweet if ch not in exclude)
    tweet = re.findall(r"[\w']+|[.,!?;]", tweet)
    # Remove hastags, names, and RT
    tweet = remove_twitter_stuff(tweet)
    return ' '.join(tweet)


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

tweet = "What’s funny is how the media makes it look like a bold and encouraging move. Then later when things don’t go right as it did for wework ceo then the same media contemplates on what went wrong."
print(clean_tweet(tweet))