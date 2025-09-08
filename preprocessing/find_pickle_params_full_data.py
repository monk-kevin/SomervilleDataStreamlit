# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 15:07:06 2025

This script performs cross-validation with an XGBClassifier on the full
Somerville Happiness Survey dataset to pickle the best parameters. These values
can then be loaded as default values within the app.py file.

@author: kevjm
"""

import pandas as pd
import os
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import pickle

path2pkl = 'C:/Users/kevjm/Documents/GitHub/SomervilleDataStreamlit/data/'
df = pd.read_pickle(path2pkl + 'df_happiness_prepped.pkl')

# classify columns as numeric or categorical
col_numeric = df.select_dtypes(include=['float64','int64']).columns.tolist()
col_categorical = df.select_dtypes(exclude=['float64','int64']).columns.tolist()

# find categorical columns that are not repeated within the numerical columns
col_categorical_unique = []
for s_cat in col_categorical:
    if (
            not any(s_cat in s_num for s_num in col_numeric) 
            and 'label' not in s_cat
            ):
        col_categorical_unique.append(s_cat)
        
        OHE_col = col_categorical_unique + ['Ward','Year']
        MMS_col =  ['CrashCount','CrimeCount',
                    'CrashCount_PerYearPerWard','CrimeCount_PerYearPerWard',
                    'bike_count','ped_count','ped2bike_foldinc',
                    'Income.Per.Number.In.Household','Rent.Mortgage.mid',
                    'Rent.Mortgage.Per.Bedroom','ACS.Somerville.Median.Income']

#define a new column transformer that removes the categorical columns
CT_dropcat = ColumnTransformer(
    transformers = [
        ('dropcat','drop',OHE_col),
        ('MinMax',MinMaxScaler(),MMS_col)
        ],
    remainder = 'passthrough'
    )   

# create pipeline to pre-process data in X for model fit
pipeline_xgb = Pipeline([
    ('encode_scale',CT_dropcat),
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('bst',XGBClassifier(objective = 'multi:softmax',
                         num_class = 5,
                         eval_metric = 'mlogloss'))
    ])

# Train / Test Split with Stratification
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Happiness.5pt.num'],axis=1),
                                                    df['Happiness.5pt.num'],
                                                    test_size = 0.2,
                                                    random_state = 42,
                                                    stratify = df['Happiness.5pt.num'])
y_train = y_train - 1
y_test = y_test - 1

# setting up for cross-validation
params = {
        'bst__eta': [0.01, 0.05, 0.1, 0.2, 0.3],
        'bst__min_child_weight': [1, 2, 4, 8, 10],
        'bst__gamma': [0, 0.75, 1.5, 3, 6],
        'bst__max_depth': [2, 4, 6, 8, 10],
        }
skf = StratifiedKFold(n_splits = 5)

XGB_CV = RandomizedSearchCV(pipeline_xgb,
                            param_distributions=params,
                            n_iter = 10,
                            scoring = "f1_weighted",
                            cv = skf.split(X_train,y_train),
                            random_state = 42)

#fitting and evaluating
XGB_CV.fit(X_train,y_train);

best_params = XGB_CV.best_params_
file_path = path2pkl + '/best_params_full.pkl'
with open(file_path,"wb") as f:
    pickle.dump(best_params, f, protocol=pickle.HIGHEST_PROTOCOL)