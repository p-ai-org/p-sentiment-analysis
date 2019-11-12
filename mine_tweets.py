import tweepy_miner_class
import pandas as pd
import datetime
import csv
import sys

def main(argv):
    """
    initializes a TweetMiner object, mines Tweets for the query
    'wework ipo', and saves the Pandas dataframe to a .csv file
    """ 

    rapid_api_key = '568b853ebamshaad79576249ddcbp1921f7jsn634271203771'

    twitter_keys = {
        'consumer_key':         'ytF42avewxOOM3sIfesKzkbAZ',
        'consumer_secret':      'ZOHNaPfPYT6K2toYqRgI2GjsXBy5klIHU8xERfXAOrLPIH6KBw',
        'access_token_key':     '1019863774101061632-ELZ22Ze3iCi12nSuVhPh60EkzTZLra',
        'access_token_secret':  'YDIWTC4XkcwKXowE00VTykZe3Jjxe8XVYnkLdy94u1ld4'
    }

    miner = tweepy_miner_class.TweetMiner(keys_dict=twitter_keys, result_limit=100)

    # generate dates to scrape on
    base = datetime.datetime.today()
    date_list = [(base - datetime.timedelta(days=x)).strftime('%Y-%m-%d') for x in range(72)]

    # open csv file to write to
    with open(argv[0], 'a') as f:

        # scrape for each day
        for i in range(len(date_list)-1):

            print('scraping for day ' + date_list[i+1])

            data = miner.mine_queried_tweets_old('wework', since=date_list[i+1], until=date_list[i], max=500)
            data.to_csv(f, sep='\t', encoding='utf-8', index=False, mode='a', header=f.tell()==0)

    print('done!')

if __name__ == '__main__':
    main(sys.argv[1:])

 
