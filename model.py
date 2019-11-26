from gensim.models import FastText
import pandas as pd

# Get labeled data
training_data = pd.read_csv("trainingandtestdata/cleaned_training_data_1.csv")
# Sentences and sentiments
sentences = training_data['text'].tolist()
sentiments = training_data['sentiment'].tolist()
# Create a new cleaner dataframe to holt text, vector, and correct sentiment
cleaned = pd.DataFrame(columns=['text','vector','sentiment'])
cleaned['text'] = sentences
cleaned['sentiment'] = sentiments

# Load pre-trained fasttext model
model = FastText.load('models/fasttext_model')

# Train a fasttext model
def train_model(fname):
    # instantiate model
    model = FastText(size=4, window=3, min_count=1)
    # add vocabulary and train
    model.build_vocab(sentences=sentences)
    model.train(sentences=sentences, total_examples=len(sentences), epochs=10)
    model.save(fname)
    return model

# Add word vectors in a sentence
def sum_vectors(sent):
    vector = [0.0, 0.0, 0.0, 0.0]
    for token in sent:
        vector += model.wv[token]
    return vector.tolist()

# Add sentence vectors to cleaned
def convert_sentences_to_vecs():
    vectors = []
    for sent in sentences:
        vectors.append(sum_vectors(sent))
    cleaned['vector'] = vectors

convert_sentences_to_vecs()
cleaned.to_csv('trainingandtestdata/training_vectors.csv')


