import tensorflow as tf
import pandas as pd
import numpy as np
import re
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt

"""
WORK IN PROGRESS
"""

# Get data
data = pd.read_csv("trainingandtestdata/spread_training_vectors_complete.csv")
# Remove quotations from vector lists (result of converting lists to csv)
# data['vector'] = data['vector'].apply(ast.literal_eval)

X_train = data.loc[:, 'v0':'v249']
target = data['sentiment']

X_train, X_valid, y_train, y_valid = train_test_split(X_train, target, random_state=0)

# Since feature vectors are sparse, we conduct PCA
# We have 111 features, will set n_components to 100, close to original num features
pca = PCA(n_components=30)
pca.fit(X_train)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of components')
# plt.ylabel('Cumulative explained variance')
# plt.show()
# 20 components seems to be where the graph plateaus
NCOMPONENTS = 250
pca = PCA(n_components=NCOMPONENTS)
# Apply PCA to X_train and X_test
X_pca_train = pca.fit_transform(X_train)
X_pca_valid = pca.fit_transform(X_valid)

print('X train:' + str(X_train.shape))
print('y train:' + str(y_train.shape))

def reshape_for_1DCNN(X):
    return np.expand_dims(X, axis=2)

X_train_CNN = reshape_for_1DCNN(X_train)
X_test_CNN = reshape_for_1DCNN(X_valid)
X_pca_train_CNN = reshape_for_1DCNN(X_pca_train)
X_pca_valid_CNN = reshape_for_1DCNN(X_pca_valid)

""" 1DCNN """

# 1DCNN function
def run_1DCNN(X_train, y_train, X_test, y_test, input_dim, kernal_size, pool_size, dropout, epochs, batch_size):
    print('-----------Running 1D CNN-----------')
    # Build model 
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=kernal_size, activation='relu', input_shape=(input_dim, 1)))
    model.add(Conv1D(filters=128, kernel_size=kernal_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
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

""" 1DCNN for GridSearch """

def grid_1DCNN(dropout_rate = 0.0):
    print('-----------Running 1D CNN-----------')
    # Build model 
    model = Sequential()
    model.add(Conv1D(filters=64, activation='relu', kernel_size=3, input_shape=(250, 1)))
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(filters=128, activation='relu', kernel_size=3))
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
    return model

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
    model.add(Dense(12, input_dim=20, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    print(model.metrics_names)
    print(model.summary())
    return model

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
# model = run_1DCNN(X_pca_train_CNN, y_train, X_pca_valid_CNN, y_valid, 20, 3, 3, 0.2, 50, 50)

model = KerasClassifier(build_fn = grid_1DCNN)
batch_size = [5, 10, 15]
epochs = [10, 15]
dropout_rate = [0.01, 0.1, 0.2]

param_grid = dict(batch_size=batch_size, epochs=epochs, dropout_rate=dropout_rate)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_pca_valid_CNN, y_valid)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
