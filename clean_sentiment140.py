import clean_tweets
import pandas as pd
data = pd.read_csv("trainingandtestdata/testdata.manual.2009.06.14.csv")
# print(data.head())
data = data.loc[:, 'text']
data = data.apply(clean_tweets.clean_tweet)
print(data.head())
data.to_csv(r'trainingandtestdata/cleanedsentiment.csv')