import tweepy
import pandas as pd
import json

auth = tweepy.OAuthHandler('CzIGg1C9kxtXtJ8PXeMNgIxW0',
                           'kpY4r4OzXT7OQme0rAugRKNJnStXyiqCXUoqi7sMa1th3dqQSA')
auth.set_access_token('921050102164283394-wFCEFzYDLxn98CGmFl39ycKe93p2VOb',
                      'tRX4RBGjXnd5Jrm9lc0ALO2Gb5waGvRdmAg439zajwCSI')

api = tweepy.API(auth)

twts = []

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text.encode("utf-8"))