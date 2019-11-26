import clean_tweets
import pandas as pd

# Get training data
data = pd.read_csv("trainingandtestdata/training_data_1.csv")
# For now, just take first 1000
data = data.loc[0:1000, ['text', 'sentiment']]
data = data[data.sentiment != 'n']
data['text'] = data['text'].apply(clean_tweets.clean_tweet)
data.to_csv(r'trainingandtestdata/cleaned_training_data_1.csv')