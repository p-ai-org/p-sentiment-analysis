import pandas as pd

''' Take the csvs from google sheets and convert them to into one dataframe '''

def combine(x):
    # If both aren't filled in, don't use it
    if isinstance(x['sentiment_a'], float) and isinstance(x['sentiment_b'], float):
        return 'n'
    # If only one is filled in, use it
    if isinstance(x['sentiment_a'], float):
        return x['sentiment_b']
    if isinstance(x['sentiment_b'], float):
        return x['sentiment_a']
    # If they disagree...
    if x['sentiment_a'] != x['sentiment_b']:
        # If one is neutral, call it neutral
        if x['sentiment_a'] == '2' or x['sentiment_b'] == '2':
            return '2'
        # One is positive and one is negative, discard
        else:
            return 'n'
    # They agree! Use either one
    return x['sentiment_a']

# Load sheets
a_sheet = pd.read_csv("trainingandtestdata/a_training_sheet.csv")
b_sheet = pd.read_csv("trainingandtestdata/b_training_sheet.csv")

# Create final sheet
final_sheet = pd.DataFrame(columns = ['text', 'sentiment'])
# Get text from either sheet
final_sheet['text'] = a_sheet['text']
final_sheet['sentiment_a'] = a_sheet['sentiment']
final_sheet['sentiment_b'] = b_sheet['sentiment']
# Combine sheets
final_sheet['sentiment'] = final_sheet.apply(combine, axis = 1)
final_sheet = final_sheet.drop('sentiment_a', axis = 1)
final_sheet = final_sheet.drop('sentiment_b', axis = 1)

print(final_sheet.head())

final_sheet.to_csv(r'trainingandtestdata/compiled_training_data.csv')