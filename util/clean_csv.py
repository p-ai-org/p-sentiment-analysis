import clean_tweets
import pandas as pd
# import numpy as np
# from numpy import load, save

''' Clean the tweets in the training data and save to a new csv '''
def clean_csv(df):
    # Get training data
    data = df
    data = data[['text', 'sentiment']]
    # Remove non-english tweets
    data = data[data.sentiment != 'n']
    # Clean tweets
    data['text'] = data['text'].apply(clean_tweets.clean_tweet)
    data.reset_index(drop=True, inplace=True)
    # Return df
    return data