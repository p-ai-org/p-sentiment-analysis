import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def clean_tweet(tweet):
    tweet = remove_unwanted(tweet)
    tokens = nltk.word_tokenize(tweet)
    tokens = remove_stop_words(tokens)
    tweet_cleaned = lemmatize(tokens)
    return tweet_cleaned

def remove_unwanted(tweet):
    tweet = re.findall(r"[\w']+|[.,!?;]", tweet)
    tweet = remove_urls(tweet)
    tweet = remove_hashtags(tweet)
    tweet = remove_usernames(tweet)
    return ' '.join(tweet)


def lemmatize(tokens):
    return [lemmatizer.lemmatize(w, get_pos(w)) for w in tokens]

def get_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

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

tweet = "Hey, how are you doing? Things are great over here, not going to lie #liars. You should check out this link: https://www.machinelearningplus.com/nlp/lemmatization-examples-python/. Thanks @StackOverflow!"
print(clean_tweet(tweet))