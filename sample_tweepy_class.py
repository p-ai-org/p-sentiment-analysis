import tweepy
import datetime
import pandas as pd


class TweetMiner(object):

    result_limit = 20
    data = []
    api = False

    twitter_keys = {
        'consumer_key':        'CzIGg1C9kxtXtJ8PXeMNgIxW0',
        'consumer_secret':     'kpY4r4OzXT7OQme0rAugRKNJnStXyiqCXUoqi7sMa1th3dqQSA',
        'access_token_key':    '921050102164283394-wFCEFzYDLxn98CGmFl39ycKe93p2VOb',
        'access_token_secret': 'tRX4RBGjXnd5Jrm9lc0ALO2Gb5waGvRdmAg439zajwCSI'
    }

    def __init__(self, keys_dict=twitter_keys, api=api, result_limit=20):

        self.twitter_keys = keys_dict

        auth = tweepy.OAuthHandler(
            keys_dict['consumer_key'], keys_dict['consumer_secret'])
        auth.set_access_token(
            keys_dict['access_token_key'], keys_dict['access_token_secret'])

        self.api = tweepy.API(auth)
        self.twitter_keys = keys_dict

        self.result_limit = result_limit

    def mine_user_tweets(self, user="dril",  # BECAUSE WHO ELSE!
                         mine_rewteets=False,
                         max_pages=5):
        data = []
        last_tweet_id = False
        page = 1

        while page <= max_pages:
            if last_tweet_id:
                statuses = self.api.user_timeline(screen_name=user,
                                                  count=self.result_limit,
                                                  max_id=last_tweet_id - 1,
                                                  tweet_mode='extended',
                                                  include_retweets=True
                                                  )
            else:
                statuses = self.api.user_timeline(screen_name=user,
                                                  count=self.result_limit,
                                                  tweet_mode='extended',
                                                  include_retweets=True)

            for item in statuses:

                mined = {
                    'tweet_id':        item.id,
                    'name':            item.user.name,
                    'screen_name':     item.user.screen_name,
                    'retweet_count':   item.retweet_count,
                    'text':            item.full_text.encode("utf-8"),
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
                except:
                    mined['retweet_text'] = 'None'
                try:
                    mined['quote_text'] = item.quoted_status.full_text
                    mined['quote_screen_name'] = status.quoted_status.user.screen_name
                except:
                    mined['quote_text'] = 'None'
                    mined['quote_screen_name'] = 'None'

                last_tweet_id = item.id
                data.append(mined)

            page += 1

        return data

miner = TweetMiner(result_limit=200)
mined_tweets = miner.mine_user_tweets(user="MarcosA73096454", max_pages=17)

mined_tweets_df = pd.DataFrame(mined_tweets)

print(mined_tweets_df['text'])