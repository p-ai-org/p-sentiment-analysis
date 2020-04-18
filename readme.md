# P-ai sentiment analysis project

## Goal

A machine learning model that can distinguish between positive, neutral and negative sentiments expressed in tweets aimed at a particular company.

## In-development usage - for members

Here's a brief description of what each file is / does:

* `trainingandtestdata`: a folder containing a number of csv files used in the training process. This folder should already have two files, `a_training_sheet.csv` and `b_training_sheet.csv` which are directly from our google sheets data
* `util/`:
    * `clean_csv`: Contains a method 'clean_csv' that applies our cleaning method to the `text` column of a specified csv
    * `clean_tweets`: Contains the method for cleaning a single tweet
    * `compile_spreadsheets`: Takes the google sheets we labeled in csv format and cleans it up into one csv
    * `download_data`: Gets the tweets from Twitter using Twitter API
    * `training_data`: Returns a pandas dataframe of our training data
    * `vectorize_data`: Applies the fast text model to vectorize tweets- methods for both averaged and non-averaged vectors
    * `streamline_data`: Runs the whole process from the google sheets csv's to usable vectorized data
* `sentiment_model`: Where we work on our classification models
* `train_fasttext`: Trains and saves a FastText model

Also note that some of the code will try to add files to or access files from certain folders that you may not have, you can just create them yourselves. Gensim models aren't pushed to GitHub because the files are too big.

## The whole 2-step process
1. Do one of the following to get the data:
    * (A) Download data from google drive, which includes csv's in `trainingandtestdata/` and numpyfiles for LSTMs in `numpyfiles/`
    * (B) Process the data yourself: this just means running the method `util/streamline_data.py`, choosing whether to average the vectors (1DCNN uses averaged vectors, LSTMs don't) and a filename
2. Run `sentiment_model`! Note that there's a lot of commented code because of the different things we've been trying- this should later be separated into different files
