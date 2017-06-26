# -*- coding: utf-8 -*-
"""
Created on Mon May  8 18:47:05 2017

@author: Babishula
"""

import pandas as pd
import numpy as np
import csv
import re
from sklearn.ensemble import *
from sklearn.preprocessing import *
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
import itertools
from operator import itemgetter

def impute_data(df, imp_col, in_terms_of):
    # Grab all the features that can be included in a Random Forest Regressor
    imp_df = df[[imp_col] + in_terms_of]

    # Split known and unknown entries into separate dataframes
    known_vals = imp_df.loc[(imp_df[imp_col].notnull())]
    unknown_vals = imp_df.loc[(imp_df[imp_col].isnull())]

    # All age values are stored in a target array
    y = known_vals.values[:, 0]

    # All the other values are stored in the feature array
    X = known_vals.values[:, 1::]

    # Create and fit a model
    rf = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rf.fit(X, y)

    # Use the fitted model to predict the missing values
    predicted_elements = rf.predict(unknown_vals.values[:, 1::])

    # Assign those predictions to the full data set
    df.loc[(df[imp_col].isnull()), imp_col] = predicted_elements
    return df


def categorize_data(df, cat_col):
    dummies = pd.get_dummies(df[cat_col])
    dummies = dummies.rename(columns=lambda cat: cat_col + '_' + str(cat))
    df = pd.concat([df, dummies], axis=1)
    return df


def bin_data(df, bin_col, bins=4):
    bin_nums = np.arange(bins)
    df[bin_col+'_bin'] = pd.qcut(df[bin_col], bins, bin_nums)
    dummies = pd.get_dummies(df[bin_col+'_bin'])
    dummies = dummies.rename(columns=lambda x: bin_col + '_' + str(x))
    df = pd.concat([df, dummies], axis=1)
    return df


def scale_data(df, scale_col):
    scaler = StandardScaler()
    new_col = str(scale_col)+'_scaled'
    df[new_col] = scaler.fit_transform(df[scale_col])
    return df


def engineer_titles(df):
    # title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
    #               'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess',
    #               'Don', 'Jonkheer']
    df['Title'] = df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
    df['Title'][df.Title == 'Jonkheer'] = 'Master'
    df['Title'][df.Title.isin(['Ms', 'Mlle'])] = 'Miss'
    df['Title'][df.Title == 'Mme'] = 'Mrs'
    df['Title'][df.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
    df['Title'][df.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'

    df = categorize_data(df, 'Title')

    # df['FamilySize'] = df['SibSp'] + df['Parch']
    # df['Age^2'] = df.Age * df.Age
    # df['Wife'] = df['Name'].apply(lambda name: 1 if 'Mrs.' in name else 0)
    # df['Scandinavian'] = df['Name'].apply(
    # lambda name: 1 if any(part in name for part in ['sson,', 'sen,', 'strom,', 'lund', 'qvist']) else 0)
    # df['Scot_irish'] = df['Name'].apply(lambda name: 1 if any(part in name for part in ['Mc', 'Mac,', 'O\',']) else 0)
    # df['Child'] = df['Age'].apply(lambda age: 1 if age < 18 else 0)
    return df


def engineer_variables(df):
    df['FamilySize'] = df['SibSp'] + df['Parch']
    return df


def transform_data(df):
    # df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    df = engineer_variables(df)

    df = categorize_data(df, 'Sex')
    df = categorize_data(df, 'Embarked')

    # Replace missing values with "U0"
    df['Cabin'][df.Cabin.isnull()] = 'U0'
    # create feature for the alphabetical part of the cabin number
    df['CabinLetter'] = df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    # convert the distinct cabin letters with incremental integer values
    df['CabinLetter'] = pd.factorize(df['CabinLetter'])[0]

    # Create a feature for the deck
    df['Deck'] = df['Cabin'].map(lambda x : re.compile("([a-zA-Z]+)").search(x).group())
    df['Deck'] = pd.factorize(df['Deck'])[0]

    # Create binary features for each deck
    decks = pd.get_dummies(df['Deck']).rename(columns=lambda x: 'Deck_' + str(x))
    df = pd.concat([df, decks], axis=1)

    # Create feature for the room number
    # df['Room'] = df['Cabin'].map( lambda x : re.compile("([0-9]+)").search(x).group()).astype(int) + 1

    df = engineer_titles(df)

    df['Fare'][np.isnan(df['Fare'])] = df['Fare'].median()
    # df = impute_data(df, 'Age', ['CabinLetter', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
    #                             'Sex_male', 'Sex_female', 'Man', 'Woman', 'Boy', 'Girl'])

    title_list = list(set(df['Title'].values))
    for t in title_list:
        avg = np.average(df[df.Title == t]['Age'].dropna())
        df.loc[((df.Age.isnull()) & (df.Title == t)), 'Age'] = avg

    df = bin_data(df, 'Fare', 4)

    df = scale_data(df, 'Age')
    df = scale_data(df, 'Fare')

    return df


# TRAINING DATA
training_df = pd.read_csv('D:/Documents/ZZ2/STAGE_BIGML/kaggle/train.csv', header=0)
training_df = transform_data(training_df)

# isolate survival column and feature names (all numeric)
feature_list = list(training_df._get_numeric_data().columns.values)
feature_list.remove('Survived')
feature_list.remove('PassengerId')
feature_list.remove('Deck_8')
train_data = training_df[['Survived'] + feature_list].values
features = train_data[0::, 1::]
target = train_data[0::, 0]

# TEST DATA
test_df = pd.read_csv('D:/Documents/ZZ2/STAGE_BIGML/kaggle/test.csv', header=0)
test_df = transform_data(test_df)

ids = test_df['PassengerId'].values

test_data = test_df[feature_list].values

forest = RandomForestClassifier(n_estimators=1000)
scores = cross_val_score(forest, features, target)
fit = forest.fit(features, target)
feature_importance = forest.feature_importances_

print('Score =', np.mean(scores), "+/-", np.std(scores))

feature_tups =[]
for x in range(len(feature_list)):
    feature_tups.append((feature_list[x], feature_importance[x]))

print(sorted(feature_tups, key=itemgetter(1), reverse=True))

output = fit.predict(test_data).astype(int)

# write random forest prediction to file
predictions_file = open("./random_forest_modek.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId", "Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()