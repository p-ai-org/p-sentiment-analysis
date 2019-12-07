import tensorflow as tf
import pandas as pd
import re
import ast
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

"""
WORK IN PROGRESS
"""

# Get data
data = pd.read_csv("trainingandtestdata/spread_training_vectors.csv")
# Remove quotations from vector lists (result of converting lists to csv)
# data['vector'] = data['vector'].apply(ast.literal_eval)

X_train = data.loc[:, 'v0':'v99']
target = data['sentiment']

X_train, X_valid, y_train, y_valid = train_test_split(X_train, target, random_state=0)

print('X train:')
print(X_train.shape)
print('y train:')
print(y_train.shape)

def reshape_for_1DCNN(X):
    return np.expand_dims(X, axis=2)
X_train = reshape_for_1DCNN(X_train)

def build_1DCNN(dropout_rate=0.0):
    print('-----------Running 1D CNN-----------')
    # Build model 
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(100, 1)))
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    # Choose optimizer and loss function
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    loss = 'sparse_categorical_crossentropy'
    # Compile 
    model.compile(optimizer=opt, 
        loss=loss,
        metrics=['acc'])
    # Fit on training data and cross-validate
    # Test on testing data
    return model

model = KerasClassifier(build_fn=build_1DCNN)
# define the grid search parameters
batch_size = [5, 10, 20]
epochs = [10, 15]
dropout_rate = [0.01, 0.1, 0.2]
param_grid = dict(batch_size=batch_size, epochs=epochs, dropout_rate=dropout_rate)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))







""" LSTM """

def create_lstm_model(embed_dim, lstm_out, batch_size):
    model = Sequential()
    model.add(Embedding(2500, embed_dim, input_length = 100, dropout = 0.2))
    model.add(LSTM(lstm_out, dropout = 0.2, recurrent_dropout = 0.2))
    model.add(Dense(1,activation='softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    print(model.summary())
    return model

""" SEQUENTIAL """

def create_simple_model():
    model = Sequential()
    model.add(Dense(12, input_dim=100, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    print(model.metrics_names)
    print(model.summary())
    return model

# model = create_lstm_model(128, 200, 32)
# model = create_simple_model()
# model.fit(X_train, y_train, batch_size = 50, epochs = 25,  verbose = 5)
# score = model.evaluate(X_valid, y_valid, 50)
# print(score)