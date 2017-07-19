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

# Define features and target values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=seed)

# feature names
full_feature_names = list(X_train)
