import pandas as pd
import ast

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