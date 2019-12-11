import clean_tweets
import pandas as pd

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