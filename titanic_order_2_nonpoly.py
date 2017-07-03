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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier

# utility
from time import time

training_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

pd.options.mode.chained_assignment = None  # default='warn'

## Set of functions to transform features into more convenient format.
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
# Function based on code from: https://www.kaggle.com/startupsci/titanic/titanic-data-science-solutions
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
test_data = combined[891:].drop('Survived', axis=1)

# transform age and fare data to have mean zero and variance 1.0
# it may only be appropriate to do a min max scaling here
scaler_pre = preprocessing.StandardScaler()
select = 'Age Fare'.split()
scale_pre = scaler_pre.fit(training_data[select])
training_data[select] = scale_pre.transform(training_data[select])

# drop uninformative data and the target feature
droplist = 'Survived PassengerId'.split()
data = training_data.drop(droplist, axis=1)

# Define features and target values
X, y = data, training_data['Survived']


# build a classifier
clf = RandomForestClassifier()

# specify parameters and distributions to sample from
param_dist = {"n_estimators": sp_randint(15, 25),
              "max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 100000
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, n_jobs=-1, cv=6)

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

# output .csv for upload
# submission = pd.DataFrame({
#         "PassengerId": iDs.astype(int),
#         "Survived": predictions.astype(int)
#     })
#
# submission.to_csv('../submission.csv', index=False)