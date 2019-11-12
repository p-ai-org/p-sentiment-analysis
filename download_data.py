import pandas as pd
import GetOldTweets3 as got

query = "WeWork"
tweets_per_day = 0 #Set to 0 to get all tweets
start_date = "2019-09-01"
end_date = "2019-11-09"

def main():
    date_range = pd.date_range(start_date, end_date)
    for date in date_range:
        next = date+pd.Timedelta(1, unit="D")
        date_string = date.date().isoformat()
        next_string = next.date().isoformat()
        print(date_string)

        data = scrape(query, date_string, next_string, tweets_per_day)
        print("\t"+str(data.size))
        data.to_csv("output/daily/day_"+date_string+".csv")

def scrape(query, since, until, max):
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(query)\
                                               .setSince(since)\
                                               .setUntil(until)\
                                               .setMaxTweets(max)
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    calculated_values = {"id": lambda x: x.id,
                        "permalink": lambda x: x.permalink,
                        "username": lambda x: x.username,
                        "to": lambda x: x.to,
                        "text": lambda x: x.text,
                        "date": lambda x: x.date,
                        "retweets": lambda x: x.retweets,
                        "favorites": lambda x: x.favorites,
                        "replies": lambda x: x.favorites,
                        "mentions": lambda x: x.mentions,
                        "hashtags": lambda x: x.hashtags,
                        "geo": lambda x: x.geo,
                        }
    raw = pd.DataFrame(data=tweets, columns=["object"])
    for value in calculated_values:
        raw[value] = raw.apply(lambda x: calculated_values[value](x["object"]), axis=1)\
            if raw.size > 0 else [] #Slightly hacky way to prevent an error if we have no data
    raw.drop(columns=["object"], inplace=True)
    return raw

if __name__ == "__main__":
    main()