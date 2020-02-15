from gensim.models import FastText
from util.training_data import get_our_training_data
from gensim.test.utils import common_texts
import pandas as pd
import numpy as np
from numpy import asarray
from numpy import save
import io
import ast

"""
Train a gensim FastText model with Sentiment140 data or load an existing model
Add average vector for each sentence in dataset
Save vectored data to another csv
"""

# Get the training data we labeled
cleaned_data = get_our_training_data('cleaned_training_data_complete.csv')

# Get sentiment 140 sentences as a list
def get_sentiment140():
    # Convert csv to pandas
    sentiment140 = pd.read_csv('trainingandtestdata/cleaned_sentiment_140.csv')
    # Don't use all of it or it takes too long
    sentiment140 = sentiment140.head(20000)
    # Get rid of quotes from csv
    sentiment140['text'] = sentiment140['text'].apply(ast.literal_eval)
    # Get sentences and sentiments
    lst = list(sentiment140['text'])
    return lst

# Train a fasttext model on sentiment140 data
def pre_train(sentiment140):
    # instantiate model - NOTE DIMENSION OF VECTOR
    model = FastText(size=100, window=3, min_count=1)
    print("[Built model]")
    # add vocabulary and train
    model.build_vocab(sentences=sentiment140)
    print("[Built vocab]")
    model.train(sentences=sentiment140, total_examples=len(sentiment140), epochs=10)
    print("[Trained model]")
    # return the model
    return model

# Train a fasttext model
def train_on_our_data(model, our_data):
    # add vocabulary and train
    model.build_vocab(sentences=our_data, update=True)
    model.train(sentences=our_data, total_examples=len(1000), epochs=model.epochs)
    # return the model
    return model

# Create a new fasttext model, trained on sentiment140
def create_model(fname):
    """ Train a new model """
    model = pre_train(get_sentiment140())
    """ Save the model """
    model.save('models/' + fname + '/gensim_' + fname)

create_model('model_6')