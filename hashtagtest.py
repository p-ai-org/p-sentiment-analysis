import tweepy
import csv
import pandas as pd

ckey = ""
csecret = ""
atoken = ""
asecret = ""

OAUTH_KEYS = {'consumer_key':'BWDvPWSmrJ18xEBTPbTkxiGm9', 'consumer_secret':'FodmJZP8RPjdG5ZJ0dz6xpNEjOFYG5LnFTjvmKKze8e0GmAO8b',
    'access_token_key':'769205140645642240-pFoG4e2EpEQft63BjruaLmLvuehRQDx', 'access_token_secret':'NvpPKD8xZKd14NxmuzONg2rAApMjYkJK5wmkyE1UGgBOk'}
auth = tweepy.OAuthHandler(OAUTH_KEYS['consumer_key'], OAUTH_KEYS['consumer_secret'])
api = tweepy.API(auth)

# Open/Create a file to append data
csvFile = open('wework.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search,q="#WeWork",count=100,
                           lang="en",
                           since="2019-06-01").items():
    print (tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])


# for tweet in tweepy.Cursor(api.search, q=('WeWork'), since='2018-01-01', until='2019-11-01').items(1000):
#     print tweet.entities.get('hashtags')
#     # print "Name:", tweet.author.name.encode('utf8')
#     # print "Screen-name:", tweet.author.screen_name.encode('utf8')
#     # print "Tweet created:", tweet.created_at
#     # print "Tweet:", tweet.text.encode('utf8')
#     # print "Retweeted:", tweet.retweeted
#     # print "Favourited:", tweet.favorited
#     # print "Location:", tweet.user.location.encode('utf8')
#     # print "Time-zone:", tweet.user.time_zone
#     # print "Geo:", tweet.geo
#     # print "//////////////////"

