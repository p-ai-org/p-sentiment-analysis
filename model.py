from gensim.models import FastText
import pandas as pd

training_data = pd.read_csv("trainingandtestdata/cleanedsentiment.csv")
sentences = training_data['text'].tolist()

def train_model():
    # instantiate model
    model = FastText(size=5, window=3, min_count=1)
    # add vocabulary and train
    model.build_vocab(sentences=sentences)
    model.train(sentences=sentences, total_examples=len(sentences), epochs=10)
    return model


model = train_model()
print(model.wv['shit'])
print(model.wv['hate'])
