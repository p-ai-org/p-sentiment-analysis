from gensim.models import FastText
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

# Get labeled data
def get_our_training_data(fname):
    # Convert csv to pandas
    training_data = pd.read_csv('trainingandtestdata/'+fname)
    # Remove quotations from text
    training_data['text'] = training_data['text'].apply(ast.literal_eval)
    # Get sentences and sentiments
    sentences = training_data['text'].tolist()
    sentiments = training_data['sentiment'].tolist()
    # Create a new cleaner dataframe to hold text, vector, and correct sentiment
    cleaned = pd.DataFrame(columns=['text','vector','sentiment'])
    cleaned['text'] = sentences
    cleaned['sentiment'] = sentiments
    return cleaned

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

# Average word vectors in a sentence
def average_vectors(sentence):
    # Initialize vector to length of 100
    average_vector = [0.0] * 100
    # Exit if empty sentence (to avoid div by 0)
    if len(sentence) == 0:
        return average_vector
    # For each token, add its vector to the average
    for token in sentence:
        average_vector += model.wv[token]
    # Divide vector by number of tokens in the sentence
    average_vector = [x / len(sentence) for x in average_vector]
    # Return vector as a list
    return average_vector

# Returns a 2d list of vectors for each word in the sentence
def full_vectors(sentence):
    vectors = []
    for token in sentence:
        vectors.append(model.wv[token])
    return vectors

# Add a column containing all the vectors to a df
def add_avg_vecs_to_df(sentences, df):
    vectors = []
    # Create a list of vectors for each sentence
    for sent in sentences:
        vectors.append(average_vectors(sent))
    # Add vectors as new column to df
    print("[Averaged vectors]")
    df['vector'] = vectors
    # Add 100 columns for the vectors
    for dim in range(250):
        cleaned_data['v'+str(dim)] = 0.0
    # Distribute vector across 100 columns
    for row in range(df.shape[0]):
        for dim in range(250):
            cleaned_data['v'+str(dim)][row] = df['vector'][row][dim]
    print("[Distributed vectors]")
    # Drop text column
    df = df.drop('text', 1)
    # Drop vector column
    df = df.drop('vector', 1)
    # Subtract 1 from sentiment column
    df['sentiment'] = df['sentiment'].apply(lambda x: x - 1)
    return df

    # Add a column containing all the vectors to a df
def add_full_vecs_to_df(sentences, df, max_len):
    # Set up 3d input array
    x = np.zeros((len(sentences), max_len, 100), dtype=np.float)
    for s, sentence in enumerate(sentences):
        for w, word in enumerate(sentence):
            vector = model.wv[word]
            for f in range(len(vector)):
                x[s, w, f] = vector[f]
    return x
    
    
""" EITHER LOAD A MODEL OR TRAIN ONE """

""" Load an existing model """
model = FastText.load('models/model_5/gensim_model_5')
print("[Model loaded]")

""" Train a new model """
# model = pre_train(get_sentiment140())
""" Save the model """
# model.save('models/model_5/gensim_model_5')

""" PRODUCE VECTORED DATAFRAME """

# Add vectors to the dataframe
sentences = cleaned_data['text']
sentiments = cleaned_data['sentiment'].to_numpy()
max_len = len(max(sentences, key=len))
vectored_data = add_full_vecs_to_df(sentences, cleaned_data, max_len)
# print(vectored_data)
save('numpyfiles/lstm_x_1.npy', vectored_data)
save('numpyfiles/lstm_y_1.npy', sentiments)

# Save our df to a csv
# vectored_data.to_csv('trainingandtestdata/spread_training_vectors_no_average.csv')