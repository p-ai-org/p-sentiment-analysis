import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sn
from numpy import load
import re
from keras import metrics
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# SMOTE
smt = SMOTE(random_state=0)

# Get data
data = pd.read_csv("trainingandtestdata/spread_training_vectors_complete.csv")

# Get training data and target (CNN)
# X = data.loc[:, 'v0':'v249']
# y = data['sentiment']

# Get training data and target (BLSTM)
X = load('numpyfiles/lstm_x_1.npy')
y = load('numpyfiles/lstm_y_1.npy')

# Split into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# original_shape = X_train.shape
# print(original_shape)
# reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
# print(reshaped.shape)
# X_train = np.reshape(reshaped, original_shape)

# SMOTE CNN DATA
# X_train, y_train = smt.fit_resample(X = X_train, y = y_train)

# Check the variance explained by the number of vectors
# def test_PCA():
#     # Since feature vectors are sparse, we conduct PCA
#     # We have 250 features, will set n_components to 200, close to original num features
#     pca = PCA(n_components=200)
#     pca.fit(X_train)
#     plt.plot(np.cumsum(pca.explained_variance_ratio_))
#     plt.xlabel('Number of components')
#     plt.ylabel('Cumulative explained variance')
#     plt.show()

# NCOMPONENTS = 100
# pca = PCA(n_components=NCOMPONENTS)
# Apply PCA to X_train and X_test
# X_train = pca.fit_transform(X_train)
# X_test = pca.fit_transform(X_test)

def reshape_for_1DCNN(X):
    return np.expand_dims(X, axis=2)

X_train_CNN = reshape_for_1DCNN(X_train)
X_test_CNN = reshape_for_1DCNN(X_test)

""" BLSTM """

def create_BLSTM():
    model = Sequential()
    model.add(SpatialDropout1D(0.3))
    # Dense (fully connected) layers
    model.add(Dense(64, activation='relu'))
    # Hidden BLSTM layer
    model.add(Bidirectional(LSTM(256, dropout = 0.3, recurrent_dropout = 0.3)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    opt = optimizers.SGD(lr=0.02)
    loss = 'sparse_categorical_crossentropy'
    # Compile 
    model.compile(optimizer=opt, loss = loss, metrics=['accuracy'])
    return model

""" 1DCNN """

# 1DCNN function
def create_1DCNN(input_dim = 250, kernel_size = 3, pool_size = 3, dropout = 0.1):
    print('-----------Running 1D CNN-----------')
    # Build model 
    model = Sequential()
    # Convolution layer
    model.add(Conv1D(filters=64, activation='relu', kernel_size=kernel_size, input_shape=(input_dim, 1)))
    # Dropout
    model.add(Dropout(dropout))
    # Convolution layer
    model.add(Conv1D(filters=256, activation='relu', kernel_size=kernel_size))
    # Pooling
    model.add(MaxPooling1D(pool_size=pool_size))
    # Flatten
    model.add(Flatten())
    # Dense (fully connected) layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    opt = optimizers.SGD(lr=0.01)
    loss = 'sparse_categorical_crossentropy'
    # Compile 
    model.compile(optimizer=opt, loss = loss, metrics=['accuracy'])
    return model

# """ 1DCNN for GridSearch """

# def grid_1DCNN(dropout_rate = 0.0):
    # print('-----------Running 1D CNN-----------')
    # # Build model 
    # model = Sequential()
    # model.add(Conv1D(filters=64, activation='relu', kernel_size=3, input_shape=(250, 1)))
    # model.add(Dropout(dropout_rate))
    # model.add(Conv1D(filters=128, activation='relu', kernel_size=3))
    # model.add(MaxPooling1D(pool_size=3))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(4, activation='softmax'))
    # # Choose optimizer and loss function
    # opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    # loss = 'sparse_categorical_crossentropy'
    # # Compile 
    # model.compile(optimizer=opt, 
    #     loss=loss,
    #     metrics=['acc'])
    # return model

""" SIMPLE SEQUENTIAL """

def create_simple_model():
    print('-----------Running Simple Sequential-----------')
    model = Sequential()
    model.add(Dense(12, input_dim=20, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    print(model.metrics_names)
    print(model.summary())
    return model

def run_simple(X_train, y_train, X_test, y_test):
    model = create_simple_model()
    model.fit(X_train, y_train, batch_size = 50, epochs = 25,  verbose = 5)
    score = model.evaluate(X_test, y_test, 50)
    print(score)

# model = KerasClassifier(build_fn = grid_1DCNN)
# batch_size = [5, 10, 15]
# epochs = [10, 15]
# dropout_rate = [0.01, 0.1, 0.2]

# param_grid = dict(batch_size=batch_size, epochs=epochs, dropout_rate=dropout_rate)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
# grid_result = grid.fit(X_train_CNN, y_train)
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

def run_BLSTM():
    print("[Building model...]")
    model = create_BLSTM()
    print("[Built model]")
    model.fit(X_train, y_train,
        epochs=5,
        batch_size=30)
    # Test on testing data
    print(model.metrics_names)
    print(model.evaluate(X_test, y_test, batch_size=20))
    return model

def run_CNN():
    print("[Building model...]")
    model = create_1DCNN()
    print("[Built model]")
    model.fit(X_train_CNN, y_train,
        epochs=3,
        batch_size=20)
    # Test on testing data
    print(model.metrics_names)
    print(model.evaluate(X_test_CNN, y_test, batch_size=20))
    return model

def BLSTM_confusion_matrix():
    model = run_BLSTM()
    y_pred = model.predict_classes(X_test)
    cm = confusion_matrix(y_test, y_pred)
    display_matrix(cm)

def CNN_confusion_matrix():
    model = run_CNN()
    y_pred = model.predict_classes(X_test_CNN)
    cm = confusion_matrix(y_test, y_pred)
    display_matrix(cm)

def display_matrix(cm):
    df_cm = pd.DataFrame(cm, range(3), range(3))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.show()

BLSTM_confusion_matrix()
# CNN_confusion_matrix()