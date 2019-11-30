import tensorflow as tf
import pandas as pd
import re
import ast
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split

"""
WORK IN PROGRESS
"""

# Get data
data = pd.read_csv("trainingandtestdata/training_vectors.csv")
# Remove quotations from vector lists (result of converting lists to csv)
data['vector'] = data['vector'].apply(ast.literal_eval)

X_train = data['vector']
target = data['sentiment']

X_train, X_valid, y_train, y_valid = train_test_split(X_train, target, random_state=0)

""" MODEL TRAINING HERE """