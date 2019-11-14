import pandas as pd
import GetOldTweets3 as got

maxTweets = 1000 #Set to 0 to get all tweets

tweetCriteria = got.manager.TweetCriteria().setQuerySearch("WeWork")\
                                           .setSince("2019-06-01")\
                                           .setUntil("2019-07-01")\
                                           .setMaxTweets(maxTweets)
tweets = got.manager.TweetManager.getTweets(tweetCriteria)

raw = pd.DataFrame(data=tweets, columns=["object"])
calculated_values = {"id": lambda x: x.id,
                     "permalink": lambda x: x.permalink,
                     "username": lambda x: x.username,
                     "to": lambda x: x.to,
                     "text": lambda x: x.text,
                     "date": lambda x: x.date,
                     "formatted_date": lambda x: x.formatted_date,
                     "retweets": lambda x: x.retweets,
                     "favorites": lambda x: x.favorites,
                     "replies": lambda x: x.favorites,
                     "mentions": lambda x: x.mentions,
                     "hashtags": lambda x: x.hashtags,
                     "geo": lambda x: x.geo,
                     }
for value in calculated_values:
    raw[value] = raw.apply(lambda x: calculated_values[value](x["object"]), axis=1)
raw.drop(columns=["object"], inplace=True) #Literally no more information to extract, so we can get rid of this now
print(raw)
print(raw["date"])
raw.to_csv("output/raw_large.csv")