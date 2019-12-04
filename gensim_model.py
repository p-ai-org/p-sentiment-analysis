from gensim.models import FastText
from gensim.test.utils import common_texts
import pandas as pd
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
cleaned_data = get_our_training_data('cleaned_training_data_1.csv')

# Get sentiment 140 sentences as a list
def get_sentiment140():
    # Convert csv to pandas
    sentiment140 = pd.read_csv('trainingandtestdata/cleaned_sentiment_140.csv')
    # Get rid of quotes from csv
    sentiment140['text'] = sentiment140['text'].apply(ast.literal_eval)
    # Get sentences and sentiments
    return list(sentiment140['text'])

# Train a fasttext model on sentiment140 data
def pre_train(sentiment140):
    # instantiate model - NOTE DIMENSION OF VECTOR (100)
    model = FastText(size=100, window=3, min_count=1)
    # add vocabulary and train
    model.build_vocab(sentences=sentiment140)
    model.train(sentences=sentiment140, total_examples=len(sentiment140), epochs=10)
    # return the model
    return model

# Train a fasttext model
def train_on_our_data(model, our_data):
    # add vocabulary and train
    model.build_vocab(sentences=our_data, update=True)
    model.train(sentences=our_data, total_examples=len(our_data), epochs=model.epochs)
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

# Add a column containing all the vectors to a df
def add_vecs_to_df(sentences, df):
    vectors = []
    # Create a list of vectors for each sentence
    for sent in sentences:
        vectors.append(average_vectors(sent))
    # Add vectors as new column to df
    df['vector'] = vectors
    df['vector'].apply(spread_vectors)
    # Drop text column
    df = df.drop('text', 1)
    # Drop vector column
    df = df.drop('vector', 1)
    # Subtract 1 from sentiment column
    df['sentiment'] = df['sentiment'].apply(lambda x: x - 1)
    return df

def spread_vectors(vector):
    for i in range(100):
        cleaned_data['v'+str(i)] = vector[i]
    
""" EITHER LOAD A MODEL OR TRAIN ONE """

""" Load an existing model """
model = FastText.load('models/model_1/gensim_model_1')

""" Train a new model """
# model = pre_train(get_sentiment140())
# model = train_on_our_data(model, cleaned_data['text'].tolist())
""" Save the model """
# model.save('models/model_1/gensim_model_1')

""" PRODUCE VECTORED DATAFRAME """

# Add vectors to the dataframe
vectored_data = add_vecs_to_df(cleaned_data['text'], cleaned_data)

# Save our df to a csv
vectored_data.to_csv('trainingandtestdata/spread_training_vectors.csv')