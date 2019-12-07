import tensorflow as tf
import pandas as pd
import re
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

"""
WORK IN PROGRESS
"""

# Get data
data = pd.read_csv("trainingandtestdata/spread_training_vectors_unlemmatized.csv")
# Remove quotations from vector lists (result of converting lists to csv)
# data['vector'] = data['vector'].apply(ast.literal_eval)

X_train = data.loc[:, 'v0':'v299']
target = data['sentiment']

X_train, X_valid, y_train, y_valid = train_test_split(X_train, target, random_state=0)

print('X train:')
print(X_train.shape)
print('y train:')
print(y_train.shape)

""" LSTM """

def create_lstm_model(embed_dim, lstm_out, batch_size):
    model = Sequential()
    model.add(Embedding(2500, embed_dim, input_length = 300, dropout = 0.2))
    model.add(LSTM(lstm_out, dropout = 0.2, recurrent_dropout = 0.2))
    model.add(Dense(1,activation='softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    print(model.summary())
    return model

""" SEQUENTIAL """

def create_simple_model():
    model = Sequential()
    model.add(Dense(12, input_dim=300, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    print(model.metrics_names)
    print(model.summary())
    return model

def create_1DCNN():
    pass

def run_Alex_MLP(X_train, y_train, X_test, y_test, input_dim, dropout, epochs, batch_size):
    print('-----------Running Multilayer Perceptron-----------')
    # Build model 
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(4, activation='softmax'))
    # Choose optimizer and loss function
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    loss = 'sparse_categorical_crossentropy'
    # Compile 
    model.compile(optimizer=opt, 
        loss=loss,
        metrics=['accuracy'])
    # Fit on training data and cross-validate
    model.fit(X_train, y_train,
        epochs=epochs,
        batch_size=batch_size)
    # Test on testing data
    score = model.evaluate(X_test, y_test, batch_size=batch_size)
    print(score)

def create_SVM():
    model = OneVsOneClassifier(LinearSVC(random_state=0))
    return model

def run_SVM():
    model = create_SVM()
    model.fit(X_train, y_train)
    prediction = model.predict(X_valid)
    score = accuracy_score(y_valid, prediction)
    print(score)

def run_simple():
    model = create_simple_model()
    model.fit(X_train, y_train, batch_size = 50, epochs = 25,  verbose = 5)
    score = model.evaluate(X_valid, y_valid, 50)
    print(score)

# run_SVM()
# run_simple()
run_Alex_MLP(X_train, y_train, X_valid, y_valid, 300, .1, 25, 50)