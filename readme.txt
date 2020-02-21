# P-ai sentiment analysis project

## Goal

A machine learning model that can distinguish between positive, neutral and negative sentiments expressed in tweets aimed at a particular company.

## In-development usage - for members

Here's a brief description of what each file is / does:

* **trainingandtestdata**: a folder containing a number of cv files used in the training process. *Note* that this folder isn't up to date because the files got too big, you'll find the data you need in the google drive foler.
* **util**:
    * **clean_csv**: Contains a method 'clean_csv' that applies our cleaning method to the 'text' column of a specified csv
    * **clean_tweets**: Contains the method for cleaning a single tweet
    * **compile_spreadsheets**: Takes the google sheets we labeled in csv format and cleans it up into one csv
    * **download_data**: Gets the tweets from Twitter using Twitter API
    * **training_data**: Returns a pandas dataframe of our training data
* **train_fasttext**: Trains a FastText model and creates 
