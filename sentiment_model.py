import tensorflow as tf
import pandas as pd
import re
import ast
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from sklearn.model_selection import train_test_split
<<<<<<< HEAD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

=======
>>>>>>> parent of b51e670... tested normal 1DD with gridsearch

"""
WORK IN PROGRESS
"""

# Get data
df = pd.read_csv("trainingandtestdata/spread_training_vectors_complete.csv")
# Remove quotations from vector lists (result of converting lists to csv)
# data['vector'] = data['vector'].apply(ast.literal_eval)

df = df.reindex(np.random.permutation(df.index))
train, test = train_test_split(df, test_size=0.10)
# X features are set to all columns excluding the label
X_train = train[train.columns.difference(['sentiment'])]
y_train = train['sentiment']
X_test = test[train.columns.difference(['sentiment'])]
y_test = test['sentiment']

def perform_smote(X_train, y_train):
    smote = SMOTE('minority')
    X_sm, y_sm = smote.fit_sample(X_train, y_train)
    return X_sm, y_sm

X_train, y_train = perform_smote(X_train, y_train)

print(len(y_train[y_train==0]))
print(len(y_train[y_train==1]))
print(len(y_train[y_train==2]))

<<<<<<< HEAD
def reshape_for_1DCNN(X):
    return np.expand_dims(X, axis=2)
X_train = reshape_for_1DCNN(X_train)
X_test = reshape_for_1DCNN(X_test)

def run_1DCNN():
    print('-----------Running 1D CNN-----------')
    # Build model 
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(250, 1)))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Choose optimizer and loss function
    opt = tf.keras.optimizers.Adam(lr=0.01, decay=1e-6)
    loss = 'sparse_categorical_crossentropy'
    # Compile 
    model.compile(optimizer=opt, 
        loss=loss,
        metrics=['acc'])
    # Fit on training data and cross-validate
    # Test on testing data
    model.fit(X_train, y_train,
        epochs=10,
        batch_size=128)
    # Test on testing data
    score = model.evaluate(X_test, y_test, batch_size=128)

run_1DCNN()





=======
""" LSTM """
>>>>>>> parent of b51e670... tested normal 1DD with gridsearch



# model = KerasClassifier(build_fn=build_1DCNN)
# define the grid search parameters
# batch_size = [100]
# epochs = [10]
# dropout_rate = [0.1]


# param_grid = dict(batch_size=batch_size, epochs=epochs, dropout_rate=dropout_rate)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
# grid_result = grid.fit(X_train, y_train)
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

<<<<<<< HEAD
# model.fit(X_train, y_train, batch_size = 50, epochs = 25,  verbose = 5)
# score = model.evaluate(X_valid, y_valid, 50)
# print(score)
=======
# model = create_lstm_model(128, 200, 32)
model = create_simple_model()
model.fit(X_train, y_train, batch_size = 50, epochs = 25,  verbose = 5)
score = model.evaluate(X_valid, y_valid, 50)
print(score)
>>>>>>> parent of b51e670... tested normal 1DD with gridsearch
