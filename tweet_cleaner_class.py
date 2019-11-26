import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import re
import string
import emoji
from geotext import GeoText 

class TweetCleaner(object):

    stopword_list = False
    stemmer = False
    data = False
    column_label = False

    def __init__(self, data, column_label):
        self.stopword_list = list(stopwords.words('english'))
        self.stemmer = SnowballStemmer('english')
        self.data = data
        self.column_label = column_label

    def clean_data(self):
        tweet_data = data.loc[column_label]
        cleaned_tweets = df.apply(clean, axis=0)
        return cleaned_tweets

    def clean(self, tweet):
        tweet = self.textify_emoji(tweet)
        tweet = self.remove_urls(tweet)
        tweet = self.remove_hashtags(tweet)
        tweet = self.remove_usernames(tweet)
        tweet = self.remove_locations(tweet)
        tweet = self.remove_punctuation(tweet)
        tweet = self.remove_stopwords(tweet)
        tweet = self.stem_text(tweet)
        return tweet 

    def remove_punctuation(self, tweet):
        return tweet.translate(str.maketrans('', '', string.punctuation))
        
    def remove_urls(self, tweet):
        http = re.sub(r'http\S+', '', tweet)
        com = re.sub(r'\S+com\S+', '', http)
        return com

    def remove_hashtags(self, tweet):
        return re.sub(r'#\S+', '', tweet)

    def remove_usernames(self, tweet):
        return re.sub(r'@\S+', '', tweet)

    def remove_locations(self, tweet):
        places = GeoText(tweet)
        words = self.tokenize(tweet)
        filtered = [word for word in words if words not in places.cities \
                                           and words not in places.countries]
        return ' '.join(filtered)

    def remove_stopwords(self, tweet):
        words = self.tokenize(tweet)
        filtered = [word for word in words if word not in self.stopword_list]
        return ' '.join(filtered)

    def textify_emoji(self, tweet):
        return emoji.demojize(tweet)

    def stem_text(self, tweet):
        words = self.tokenize(tweet)
        stemmed = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed)

    def tokenize(self, tweet):
        return re.split('W+', tweet)


def main():
    sample_tweet = '#TheBigDay out everywhere now http://chanceraps.com & get tix \
                    to the tour @chancetherapper San Francisco'

    cleaner = TweetCleaner(None, None)
    print('original: ' + sample_tweet)
    print('cleaned: ' + cleaner.clean(sample_tweet))

if __name__ == '__main__':
    main()
    
