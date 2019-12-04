import pandas as pd
import GetOldTweets3 as got #See https://pypi.org/project/GetOldTweets3/

query = "WeWork"
tweets_per_day = 0 #Set to 0 to get all tweets
start_date = "2019-09-01"
end_date = "2019-11-09"

def main():
    sum_tweets = 0
    date_range = pd.date_range(start_date, end_date)
    print("Starting download.")
    if tweets_per_day > 0: print("\tLimiting to "+str(tweets_per_day)+" tweets per day.")
    else: print("\tNo per-day limit.")
    for date in date_range:
        next = date+pd.Timedelta(1, unit="D")
        date_string = date.date().isoformat() #GetOldTweets3 takes dates as strings
        next_string = next.date().isoformat()
        print(date_string)
        data = scrape(query, date_string, next_string, tweets_per_day, retry=5)
        print("\t"+str(len(data))+" tweets on this day")
        sum_tweets += len(data)
        print("\t"+str(sum_tweets)+" total tweets so far")
        data.to_csv("output/daily/day_"+date_string+".csv") #Save a file for each day so that if days fail or we expand the timeframe we don't have to start over
    print("Done.")
    print("Downloaded "+str(sum_tweets)+" tweets across "+str(len(date_range))+" days")

def scrape(query, since, until, max, retry=1): #Use GetOldTweets3 to query Twitter and return the results in a Pandas DataFrame
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(query)\
                                               .setSince(since)\
                                               .setUntil(until)\
                                               .setMaxTweets(max) #Tell the library what to search for
    tweets = None
    for attempt in range(retry): #Sometimes attempts return zero tweets and simply retrying will fix it. Not sure why this is happening.
        #TODO: a timeout for this would be nice
        tweets = got.manager.TweetManager.getTweets(tweetCriteria) #This is the part that actually downloads the data
        if len(tweets) > 0: break
        else: print("\tAttempt "+str(attempt)+" got 0 tweets. "+"Retrying." if attempt < retry-1 else "Moving on.")
    calculated_values = {"id": lambda x: x.id,
                        "permalink": lambda x: x.permalink,
                        "username": lambda x: x.username,
                        "author_id": lambda x: x.author_id,
                        "text": lambda x: x.text,
                        "date": lambda x: x.date,
                        "formatted_date": lambda x: x.formatted_date,
                        "to": lambda x: x.to,
                        "retweets": lambda x: x.retweets,
                        "favorites": lambda x: x.favorites,
                        "replies": lambda x: x.favorites,
                        "mentions": lambda x: x.mentions,
                        "hashtags": lambda x: x.hashtags,
                        "urls": lambda x: x.urls,
                        "geo": lambda x: x.geo,
                        } #All the variables we care about and how to find them given a Tweet object
    raw = pd.DataFrame(data=tweets, columns=["object"])
    for value in calculated_values: #Iterate over all the variables we want to find; next line automatically iterates over all Tweets
        raw[value] = raw.apply(lambda x: calculated_values[value](x["object"]), axis=1)\
            if raw.size > 0 else [] #Slightly hacky way to prevent an error if we have no data
    raw.drop(columns=["object"], inplace=True) #No need to keep the objects once we've extracted all the data
    return raw

if __name__ == "__main__":
    main()