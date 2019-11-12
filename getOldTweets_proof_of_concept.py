import GetOldTweets3 as got

tweetCriteria = got.manager.TweetCriteria().setQuerySearch('wework')\
                                           .setSince("2019-06-01")\
                                           .setUntil("2019-10-31")\
                                           .setMaxTweets(10)
tweets = got.manager.TweetManager.getTweets(tweetCriteria)
for tweet in tweets:
    print(tweet.text)