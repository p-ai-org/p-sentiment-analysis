from gensim.models import FastText
from training_data import get_our_training_data
import numpy as np
from numpy import save
import pandas as pd

""" PRODUCE VECTORED DATAFRAME """

# Add a column containing all the vectors to a df
def add_full_vecs_to_df(sentences, df, max_len, model, vector_size):
    # Set up 3d input array
    x = np.zeros((len(sentences), max_len, vector_size), dtype=np.float)
    for s, sentence in enumerate(sentences):
        for w, word in enumerate(sentence):
            vector = model.wv[word]
            for f in range(len(vector)):
                x[s, w, f] = vector[f]
    return x

# Average word vectors in a sentence
def average_vector(sentence, model, vector_size):
    ''' NOT USED FOR LSTMs '''
    # Initialize vector to length of 100
    average_vector = [0.0] * vector_size
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

def save_averaged_vectored_data(df, vector_size, model_name, fname):
    # Get data
    cleaned_data = df
    # Load model
    model = FastText.load('models/' + model_name + '/gensim_' + model_name)
    # Add 100 columns for the vectors
    for dim in range(vector_size):
        cleaned_data['v' + str(dim)] = 0.0
    # Prepare vectors
    vectors = []
    # Create a list of vectors for each sentence
    for sent in cleaned_data['text']:
        vectors.append(average_vector(sent, model, vector_size))
    print("  [Averaged vectors]")
    # Distribute vectors to individual columns
    for row in range(cleaned_data.shape[0]):
        for dim in range(vector_size):
            cleaned_data['v' + str(dim)][row] = vectors[row][dim]
    print("  [Distributed vectors]")
    # Drop text column
    cleaned_data = cleaned_data.drop('text', 1)
    # Save data
    cleaned_data.to_csv(r'trainingandtestdata/' + fname + '.csv')

''' Vectorize cleaned data into two 3d numpy arrays (X and y) '''
def save_vectored_data(df, vector_size, model_name, fname):
    # Get data
    cleaned_data = df
    # Load model
    model = FastText.load('models/' + model_name + '/gensim_' + model_name)
    # Add vectors to the dataframe
    sentences = cleaned_data['text']
    sentiments = cleaned_data['sentiment'].to_numpy(dtype = np.int)
    max_len = len(max(sentences, key=len))
    vectored_data = add_full_vecs_to_df(sentences = sentences, df = cleaned_data, max_len = max_len, model = model, vector_size = vector_size)
    # Subtract 1 from sentiment column (1-3 --> 0-2)
    for i in range(sentiments.shape[0]):  # iterate over rows
        sentiments[i] = sentiments[i] - 1

    # Save vectored data
    save('numpyfiles/'+ fname +'_X.npy', vectored_data)
    save('numpyfiles/'+ fname +'_y.npy', sentiments)