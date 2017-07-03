# data analysis and wrangling
import pandas as pd
import numpy as np
import scipy

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from sklearn import preprocessing
import fancyimpute
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import xgboost as xgb

# utility
from time import time

training_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# Set of functions to transform features into more convenient format.
#
# Code performs three separate tasks:
#   (1). Pull out the first letter of the cabin feature.
#          Code taken from: https://www.kaggle.com/jeffd23/titanic/scikit-learn-ml-from-start-to-finish
#   (2). Add column which is binary variable that pertains
#        to whether the cabin feature is known or not.
#        (This may be relevant for Pclass = 1).
#   (3). Recasts cabin feature as number.
def simplify_cabins(data):
    data.Cabin = data.Cabin.fillna('N')
    data.Cabin = data.Cabin.apply(lambda x: x[0])

    cabin_mapping = {'N': 0, 'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1,
                     'F': 1, 'G': 1, 'T': 1}
    data['Cabin_Known'] = data.Cabin.map(cabin_mapping)

    le = preprocessing.LabelEncoder().fit(data.Cabin)
    data.Cabin = le.transform(data.Cabin)

    return data


# Recast sex as numerical feature.
def simplify_sex(data):
    sex_mapping = {'male': 0, 'female': 1}
    data.Sex = data.Sex.map(sex_mapping).astype(int)

    return data


# Recast port of departure as numerical feature.
def simplify_embark(data):
    # Two missing values, assign the most common port of departure.
    data.Embarked = data.Embarked.fillna('S')

    le = preprocessing.LabelEncoder().fit(data.Embarked)
    data.Embarked = le.transform(data.Embarked)

    return data


# Extract title from names, then assign to one of five ordinal classes.
# Function based on code from:
# https://www.kaggle.com/startupsci/titanic/titanic-data-science-solutions
def add_title(data):
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    data.Title = data.Title.replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
                                     'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data.Title = data.Title.replace('Mlle', 'Miss')
    data.Title = data.Title.replace('Ms', 'Miss')
    data.Title = data.Title.replace('Mme', 'Mrs')

    # Map from strings to ordinal variables.
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    data.Title = data.Title.map(title_mapping)
    data.Title = data.Title.fillna(0)

    return data


# Drop all unwanted features (name, ticket).
def drop_features(data):
    return data.drop(['Name', 'Ticket'], axis=1)

# Utility function to report best scores


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# Perform all feature transformations.
def transform_all(data):
    data = simplify_cabins(data)
    data = simplify_sex(data)
    data = simplify_embark(data)
    data = add_title(data)
    data = drop_features(data)

    return data


training_data = transform_all(training_data)
test_data = transform_all(test_data)
# Impute single missing 'Fare' value with median
training_data['Fare'] = training_data['Fare'].fillna(training_data['Fare'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

all_data = [training_data, test_data]
combined = pd.concat(all_data)


def impute_ages(data):
    drop_survived = data.drop(['Survived'], axis=1)
    column_titles = list(drop_survived)
    mice_results = fancyimpute.MICE().complete(np.array(drop_survived))
    results = pd.DataFrame(mice_results, columns=column_titles)
    results['Survived'] = list(data['Survived'])
    return results


combined = impute_ages(combined)

training_data = combined[:891]
test_data = combined[891:]

# transform age and fare data to have mean zero and variance 1.0
scaler = preprocessing.StandardScaler()
select = 'Age Fare'.split()
training_data[select] = scaler.fit_transform(training_data[select])
test_data[select] = scaler.transform(test_data[select])

# drop uninformative data and the target feature
droplist = 'Survived PassengerId'.split()
data_train = training_data.drop(droplist, axis=1)
data_test = test_data.drop(droplist, axis=1)

# Define features and target values
X_train, X_test, y_train, y_test = data_train, data_test, training_data['Survived'], test_data['Survived']

# feature names
full_feature_names = list(X_train)

# build xgboost model
dtrain_all = xgb.DMatrix(X_train, y_train, feature_names=full_feature_names)
dtest = xgb.DMatrix(X_test, feature_names=full_feature_names)

xgb_params = {
    'eta': 0.01,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.65,
    'objective': 'binary:logistic',
    'eval_metric': 'error',
    'min_child_weight': 4,
    'silent': 1,
    'seed': 0
}

# Perform cross-validation on training set.
cv_output = xgb.cv(xgb_params, dtrain_all, num_boost_round=1000, early_stopping_rounds=100,
                   verbose_eval=50, show_stdv=False)
num_boost_round = len(cv_output)
print(num_boost_round)

model = xgb.train(dict(xgb_params, silent=0), dtrain_all, num_boost_round=num_boost_round)


# Plot feature importance.
# fig, ax = plt.subplots(1, 1, figsize=(8, 16))
# xgb.plot_importance(model, height=0.2, ax=ax)

# Predict for test set.
y_pred = model.predict(dtest)

df_sub = pd.DataFrame(
    {"PassengerId": test_data['PassengerId'].astype(int), "Survived": y_pred.astype(int)})

df_sub.to_csv('xgboost_initial_niave.csv', index=False)
