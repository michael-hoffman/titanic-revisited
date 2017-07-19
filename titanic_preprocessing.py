#
# Set of functions used for preprocessing Titanic data. 
#
# Author: Charlie Bonfield
# Last Modified: July 2017

# Import statements
import fancyimpute
import numpy as np
import pandas as pd
#from sklearn import preprocessing

class Useful_Preprocessing(object):
    
    def one_hot_encode(self, x, n_classes):
       #One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
       #: x: List of sample Labels
       #: return: Numpy array of one-hot encoded labels
       return np.eye(n_classes)[x] 
    
    # Perform one hot encoding for all categorical variables, leaving numerical
    # variables alone.
    def all_ohe(self, data):
        survived = data['Survived']
        data.drop(['Survived'], axis=1, inplace=True)
        df_numeric = data.select_dtypes(exclude=['object'])
        df_objects = data.select_dtypes(include=['object']).copy()
        
        for column in df_objects:
            factorized_df = pd.factorize(df_objects[column])
            f_values = factorized_df[0]
            f_labels = list(factorized_df[1])
            f_extlabels = [column + '_' + s for s in f_labels]
            n_classes = len(f_labels)
            
            one_hot_encoded_features = self.one_hot_encode(f_values, n_classes)
            ohe_features_df = pd.DataFrame(one_hot_encoded_features, columns=f_extlabels)
            df_objects.drop(column, axis=1, inplace=True)
            with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
                print(ohe_features_df)
                print(df_objects)
            df_objects = pd.concat([df_objects, ohe_features_df], axis=1)
            
        all_data = pd.concat([df_numeric, df_objects], axis=1)
        all_data['Survived'] = survived
        return all_data
    
    # Code performs three separate tasks:
    #   (1). Pull out the first letter of the cabin feature.
    #          Code taken from: https://www.kaggle.com/jeffd23/titanic/scikit-learn-ml-from-start-to-finish
    #   (2). Add column which is binary variable that pertains
    #        to whether the cabin feature is known or not.
    #        (This may be relevant for Pclass = 1).
    #   (3). Recasts cabin feature as number.
    def simplify_cabins(self, data):
        data.Cabin = data.Cabin.fillna('N')
        data.Cabin = data.Cabin.apply(lambda x: x[0])

        cabin_mapping = {'N': 0, 'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1,
                         'F': 1, 'G': 1, 'T': 1}
        data['Cabin_Known'] = data.Cabin.map(cabin_mapping)

        # UNCOMMENT BELOW FOR LABEL ENCODING
        #le = preprocessing.LabelEncoder().fit(data.Cabin)
        #data.Cabin = le.transform(data.Cabin)

        return data

    # Recast sex as numerical feature.
    def simplify_sex(self, data):
        sex_mapping = {'male': 0, 'female': 1}
        data.Sex = data.Sex.map(sex_mapping).astype(int)

        return data
    
    # Recast passenger class as string (easier for OHE).
    def stringify_pclass(self, data):
        pclass_mapping = {1.0: 'U', 2.0:'M', 3.0:'L'}
        data.Pclass = data.Pclass.map(pclass_mapping)
        return data
    
    # Create new feature 'Family_Size', taken as the sum of parents/children
    # and siblings/spouses. Also create 'Is_Alone'. 
    def family_size(self, data):
        data['Family_Size'] = data['SibSp'] + data['Parch'] + 1
        data['Is_Alone'] = data['Family_Size'].apply(lambda x: 1 if x == 1 else 0)
        return data
    
    # Recast port of departure as numerical feature.
    def simplify_embark(self, data):
        # Two missing values, assign the most common port of departure.
        data.Embarked = data.Embarked.fillna('S')

        # UNCOMMENT BELOW FOR LABEL ENCODING.
        #le = preprocessing.LabelEncoder().fit(data.Embarked)
        #data.Embarked = le.transform(data.Embarked)

        return data

    # Extract title from names, then assign to one of five ordinal classes.
    # Function based on code from: https://www.kaggle.com/startupsci/titanic/titanic-data-science-solutions
    def add_title(self, data):
        data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        data.Title = data.Title.replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
                                         'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        data.Title = data.Title.replace('Mlle', 'Miss')
        data.Title = data.Title.replace('Ms', 'Miss')
        data.Title = data.Title.replace('Mme', 'Mrs')

        # UNCOMMENT BELOW FOR LABEL ENCODING.
        # Map from strings to ordinal variables.
        #title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

        #data.Title = data.Title.map(title_mapping)
        #data.Title = data.Title.fillna(0)

        return data

    # Drop all unwanted features (name, ticket).
    def drop_features(self, data):
        return data.drop(['Name', 'Ticket'], axis=1)
    
    # Perform all feature transformations.
    def transform_all(self, data):
        data = self.simplify_cabins(data)
        data = self.simplify_sex(data)
        data = self.stringify_pclass(data)
        data = self.family_size(data)
        data = self.simplify_embark(data)
        data = self.add_title(data)
        data = self.drop_features(data)
        data = self.all_ohe(data)
        return data
    
    # Impute missing ages using MICE.
    def impute_ages(self, data):
        #drop_survived = data.drop(['Survived'], axis=1)
        column_titles = list(data)
        mice_results = fancyimpute.MICE().complete(np.array(data))
        results = pd.DataFrame(mice_results, columns=column_titles)
        #results['Survived'] = list(data['Survived'])
        return results