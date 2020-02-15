import clean_tweets
import pandas as pd
# import numpy as np
# from numpy import load, save

''' Clean the tweets in the training data and save to a new csv '''

# Get training data
data = pd.read_csv("trainingandtestdata/compiled_training_data.csv")
data = data[['text', 'sentiment']]
# Remove non-english tweets
data = data[data.sentiment != 'n']
# Clean tweets
data['text'] = data['text'].apply(clean_tweets.clean_tweet)
data.reset_index(drop=True, inplace=True)
# Save data to working directory
data.to_csv(r'trainingandtestdata/cleaned_training_data_complete.csv')

# y = load('numpyfiles/lstm_y_1.npy')
# for i in range(y.shape[0]):  # iterate over rows
#     y[i] = y[i] - 1
# # np.apply_along_axis(lambda x: np.subtract(x, 1), 0, y)
# save('numpyfiles/lstm_y_1.npy', y)