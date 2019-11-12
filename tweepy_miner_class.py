import datetime
import tweepy
import pandas as pd
import GetOldTweets3 as got

class TweetMiner(object):

    result_limit   =   20
    data           =   []
    api            =   False

    def __init__(self, keys_dict, api=api, result_limit=20):
        """
        initializes a TweetMiner object and handles authentication for the app

        @params keys_dict:      keys and access tokens for Twitter API
                api:            a Tweepy API object
                result_limit:   number of tweets collected
        """

        self.twitter_keys = keys_dict

        auth = tweepy.OAuthHandler(keys_dict['consumer_key'],
                                   keys_dict['consumer_secret'])
        auth.set_access_token(keys_dict['access_token_key'],
                              keys_dict['access_token_secret'])

        self.api = tweepy.API(auth)
        self.twitter_keys = keys_dict

        self.result_limit = result_limit


    def mine_user_tweets(self, user='dril', mine_retweets=False, max_pages=5):
        """
        mines the Tweets on a given user's timeline

        @params user:            Twitter username
                mine_retweets:   true if retweets should be mined
                max_pages:       number of pages to mine tweets from

        @returns a Pandas dataframe of Tweets/Tweet information
        """
        data            =   []
        last_tweet_id   =   False
        page            =   1

        while page <= max_pages:
            if last_tweet_id:
                statuses   =   self.api.user_timeline(screen_name=user,
                                                     count=self.result_limit,
                                                     max_id=last_tweet_id - 1,
                                                     tweet_mode='extended',
                                                     include_retweets=True
                                                    )
            else:
                statuses   =   self.api.user_timeline(screen_name=user,
                                                        count=self.result_limit,
                                                        tweet_mode='extended',
                                                        include_retweets=True)

            for item in statuses:

                mined = {
                    'tweet_id':        item.id,
                    'name':            item.user.name,
                    'screen_name':     item.user.screen_name,
                    'retweet_count':   item.retweet_count,
                    'text':            item.full_text,
                    'mined_at':        datetime.datetime.now(),
                    'created_at':      item.created_at,
                    'favourite_count': item.favorite_count,
                    'hashtags':        item.entities['hashtags'],
                    'status_count':    item.user.statuses_count,
                    'location':        item.place,
                    'source_device':   item.source
                }

                try:
                    mined['retweet_text'] = item.retweeted_status.full_text
                except tweepy.TweepError as e:
                    mined['retweet_text'] = 'None'
                    print(f"Error: Retweet Not Found - \n{str(e)}")
                try:
                    mined['quote_text'] = item.quoted_status.full_text
                    mined['quote_screen_name'] = item.quoted_status.user.screen_name
                except tweepy.TweepError as e:
                    mined['quote_text'] = 'None'
                    mined['quote_screen_name'] = 'None'
                    print(f"Error: Retweet Not Found - \n{str(e)}")

                last_tweet_id = item.id
                data.append(mined)

            page += 1

        return pd.DataFrame(data)


    def mine_queried_tweets(self, query, mine_retweets=False, max_pages=5):
        """
        mines Tweets specified by a given query

        @params query:           string of keywords to search for
                mine_retweets:   true if retweets should be mined
                max_pages:       number of pages to mine Tweets from 

        @returns a Pandas dataframe of Tweets/Tweet information
        """
        data = []
        last_tweet_id = False
        page = 1

        while page <= max_pages:
            # get SearchResults objects containing the query
            if last_tweet_id:




                queried_search = self.api.search(q=query,
                                                 count=self.result_limit,
                                                 max_id=last_tweet_id-1,
                                                 lang='en',
                                                 since='2019-08-01',
                                                 until='2019-11-01',
                                                 tweet_mode='extended')

            else:
                queried_search = self.api.search(q=query,
                                                 count=self.result_limit,
                                                 lang='en',
                                                 since='2019-08-01',
                                                 until='2019-11-01',
                                                 tweet_mode='extended')

            # mine data from the SearchResults object
            for item in queried_search:

                mined = {
                    'tweet_id':        item.id,
                    'name':            item.user.name,
                    'screen_name':     item.user.screen_name,
                    'retweet_count':   item.retweet_count,
                    'text':            item.full_text,
                    'mined_at':        datetime.datetime.now(),
                    'created_at':      item.created_at,
                    'favourite_count': item.favorite_count,
                    'hashtags':        item.entities['hashtags'],
                    'status_count':    item.user.statuses_count,
                    'location':        item.place,
                    'source_device':   item.source,
                }

                if item.place:
                    mined['coordinates']:     item.place.bounding_box.coordinates

                if mine_retweets:
                    try:
                        mined['retweet_text'] = item.retweeted_status.full_text
                    except tweepy.TweepError as e:
                        mined['retweet_text'] = 'None'
                        print(f"Error: Retweet Not Found - \n{str(e)}")
                    try:
                        mined['quote_text'] = item.quoted_status.full_text
                        mined['quote_screen_name'] = item.quoted_status.user.screen_name
                    except tweepy.TweepError as e:
                        mined['quote_text'] = 'None'
                        mined['quote_screen_name'] = 'None'
                        print(f"Error: Retweet Not Found - \n{str(e)}")

                last_tweet_id = item.id
                data.append(mined)

            page += 1

        return pd.DataFrame(data)


    def mine_queried_tweets_old(self, query, since='2019-11-01', until='2019-11-07', max=100):
        """
        mines old Tweets by query using GetOldTweets3

        @params query:   string of keywords to search for
                since:   lower bound date to start mining Tweets from
                until:   upper bound date to end mining Tweets from 
                max:     maximum number of Tweets to mine

        @returns a Pandas dataframe of Tweets/Tweet information
        """
        data = []

        # set the parameters for mining Tweets
        tweet_criteria = got.manager.TweetCriteria().setQuerySearch(query)\
                                                    .setSince(since)\
                                                    .setUntil(until)\
                                                    .setMaxTweets(max)\
                                                    .setLang('en')\

        # get GOT manager containing the Tweets 
        tweets = got.manager.TweetManager.getTweets(tweet_criteria)

        # mine data from the Tweets 
        for item in tweets:

            mined = {
                'tweet_id':        item.id,
                'name':            item.username,
                'text':            item.text,
                'mined_at':        datetime.datetime.now(),
                'created_at':      item.date,
                'retweets':        item.retweets,
                'favorites':       item.favorites,
                'hashtags':        item.hashtags,
                'mentions':        item.mentions,
                'location':        item.geo,
            }

            data.append(mined)

        return pd.DataFrame(data)
