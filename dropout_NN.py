#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:03:22 2017

@author: mhoffman
"""

import pandas as pd
import numpy as np

# processing
from sklearn import preprocessing
import fancyimpute
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# ANN
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Activation, Dropout
from keras import optimizers

training_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
seed = 42

X = pd.read_csv('data_train.csv')
y = pd.read_csv('data_target.csv')

p_set = pd.read_csv('data_test.csv')

# feature names
full_feature_names = list(X)

X = X.as_matrix()

m = Sequential()
m.add(Dense(128, activation='relu', input_shape=X.shape)
m.add(Dropout(0.5))
m.add(Dense(128, activation='relu'))
m.add(Dropout(0.5))
m.add(Dense(128, activation='relu'))
m.add(Dropout(0.5))
m.add(Dense(len(np.unique(y)), activation='softmax'))

m.compile(
    optimizer=optimizers.Adam(lr=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

m.fit(
    # Feature matrix
    X,
    # Target class one-hot-encoded
    y['Survived'].as_matrix(),
    # Iterations to be run if not stopped by EarlyStopping
    epochs=200,
    callbacks=[
        # Stop iterations when validation loss has not improved
        EarlyStopping(monitor='val_loss', patience=25),
        # Nice for keeping the last model before overfitting occurs
        ModelCheckpoint(
            'best.model',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ],
    verbose=2,
    validation_split=0.1,
    batch_size=256,
)

# Load the best model
m.load_weights("best.model")

# Keep track of what class corresponds to what index
mapping = (
    pd.get_dummies(pd.DataFrame(y_train), columns=[0], prefix='', prefix_sep='')
    .columns.astype(int).values
)
y_test_preds = [mapping[pred] for pred in m.predict(X_test).argmax(axis=1)]

print(
pd.crosstab(
    pd.Series(y_test, name='Actual'),
    pd.Series(y_test_preds, name='Predicted'),
    margins=True
)
)
