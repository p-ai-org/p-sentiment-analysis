import clean_tweets
import pandas as pd
# import numpy as np
# from numpy import load, save

# Get training data
data = pd.read_csv("trainingandtestdata/compiled_training_data.csv")
# data = pd.read_csv("trainingandtestdata/sentiment_140.csv")
# For now, just take first 1000
# data = data[['text']]
data = data[['text', 'sentiment']]
data = data[data.sentiment != 'n']
# print(data.head())
data['text'] = data['text'].apply(clean_tweets.clean_tweet)
data.reset_index(drop=True, inplace=True)
data.to_csv(r'trainingandtestdata/cleaned_training_data_complete.csv')

# y = load('numpyfiles/lstm_y_1.npy')
# for i in range(y.shape[0]):  # iterate over rows
#     y[i] = y[i] - 1
# # np.apply_along_axis(lambda x: np.subtract(x, 1), 0, y)
# save('numpyfiles/lstm_y_1.npy', y)