# P-ai sentiment analysis project

## Goal

A machine learning model that can distinguish between positive, neutral and negative sentiments expressed in tweets aimed at a particular company.

## In-development usage - for members

Here's a brief description of what each file is / does:

* `trainingandtestdata`: a folder containing a number of csv files used in the training process. Note that this folder isn't up to date because the files got too big, you'll find the data you need in the google drive foler.
* `util/`:
    * `clean_csv`: Contains a method 'clean_csv' that applies our cleaning method to the `text` column of a specified csv
    * `clean_tweets`: Contains the method for cleaning a single tweet
    * `compile_spreadsheets`: Takes the google sheets we labeled in csv format and cleans it up into one csv
    * `download_data`: Gets the tweets from Twitter using Twitter API
    * `training_data`: Returns a pandas dataframe of our training data
* `sentiment_model`: Where we work on our classification models
* `train_fasttext`: Trains and saves a FastText model
* `vectorize_data`: Applies the fast text model to vectorize tweets- methods for both averaged and non-averaged vectors

Also note that some of the code will try to add files to or access files from certain folders that you may not have, you can just create them yourselves. Gensim models aren't pushed to GitHub because the files are too big.

## The whole 2-step process (which should be better automated- will do later)
1. Do one of the following to get the data:
    * (A) Download data from google drive, which includes both csv's in `trainingandtestdata/` and numpyfiles for LSTMs in `numpyfiles/`
    * (B) Process the data yourself:
        1. Use `compile_spreadsheets` to combine two files, `trainingandtestdata/a_training_sheet.csv` and `trainingandtestdata/b_training_sheet.csv` into a prettier csv file, `trainingandtestdata/compiled_training_data.csv`.
        2. Run `clean_csv` to apply the cleaning method on the `text` column of the csv
        3. Use `vectorize_data` to use a gensim model to create either a csv or numpy file of the vectors and their corresponsind sentiment
            * If you don't have any models, run `train_fasttext` which should train and save a new gensim model into a new folder
2. Run `sentiment_model`! Note that there's a lot of commented code because of the different things we've been trying- mental note to separate the models into different files perhaps
