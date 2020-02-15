from gensim.models import FastText
from util.training_data import get_our_training_data
import numpy as np
from numpy import save
import pandas as pd
import ast

""" PRODUCE VECTORED DATAFRAME """

# Get the training data we labeled
cleaned_data = get_our_training_data('cleaned_training_data_complete.csv')

# Add a column containing all the vectors to a df
def add_full_vecs_to_df(sentences, df, max_len, model):
    # Set up 3d input array
    x = np.zeros((len(sentences), max_len, 100), dtype=np.float)
    for s, sentence in enumerate(sentences):
        for w, word in enumerate(sentence):
            vector = model.wv[word]
            for f in range(len(vector)):
                x[s, w, f] = vector[f]
    return x

# Average word vectors in a sentence
def average_vectors(sentence, model):
    ''' NOT USED FOR LSTMs '''
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

# Add a column containing all the vectors to a df
def add_avg_vecs_to_df(sentences, df, model):
    ''' NOT USED FOR LSTMs '''
    vectors = []
    # Create a list of vectors for each sentence
    for sent in sentences:
        vectors.append(average_vectors(sent, model))
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

def save_vectored_data(model_name, fname):
    # Load model
    model = FastText.load('models/' + model_name + '/gensim_' + model_name)
    print("[Model loaded]")

    # Add vectors to the dataframe
    sentences = cleaned_data['text']
    sentiments = cleaned_data['sentiment'].to_numpy()
    max_len = len(max(sentences, key=len))
    vectored_data = add_full_vecs_to_df(sentences, cleaned_data, max_len, model)

    # Save vectored data
    save('numpyfiles/'+ fname +'_X.npy', vectored_data)
    save('numpyfiles/'+ fname +'_y.npy', sentiments)

save_vectored_data('model_5', 'lstm_1')