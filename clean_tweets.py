import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from geotext import GeoText
stop_words = set(stopwords.words('english'))

def clean_tweet(tweet):
    tweet_cleaned = tweet.split(' ')
    tweet_cleaned = remove_stop_words(tweet_cleaned)
    tweet_cleaned = remove_urls(tweet_cleaned)
    tweet_cleaned = remove_hashtags(tweet_cleaned)
    tweet_cleaned = remove_usernames(tweet_cleaned)
    tweet_cleaned = remove_places(tweet_cleaned)
    tweet_compiled = ' '.join(tweet_cleaned)
    tweet_compiled = word_tokenize(tweet_compiled)
    return tweet_compiled
    
def remove_hashtags(tweet_tokens):
    regex = re.compile(r'^#.+')
    return [i for i in tweet_tokens if not regex.search(i)]

def remove_usernames(tweet_tokens):
    regex = re.compile(r'^@.+')
    return [i for i in tweet_tokens if not regex.search(i)]

def remove_stop_words(tweet_tokens):
    return [w for w in tweet_tokens if not w in stop_words]

def remove_urls(tweet_tokens):
    regex = re.compile(r'^http|.*\.com.*')
    return [i for i in tweet_tokens if not regex.search(i)]

def remove_places(tweet_tokens):
    return tweet_tokens

# print(clean_tweet("Hey, how was London? I know how much you loved Las Vegas"))
# print(clean_tweet("Hey, how are you doing? My ass is tight as all hell."))
print(clean_tweet("Hey, how are you doing? This was such a nice day, and I love London. #Love. Here's a link: life.com. Thanks @Donald!"))