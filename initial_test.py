import tweepy

consumer_key, consumer_secret, access_token, access_token_secret = open("../twitter_keys.txt", "r").read().splitlines()

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
print(api.me())

public_tweets = api.home_timeline()

for tweet in public_tweets:
    print(tweet.text)