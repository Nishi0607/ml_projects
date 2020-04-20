# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 00:39:56 2020

@author: NK
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

credit_df = pd.read_csv('German Credit Data.csv')

credit_df.head()

credit_df.info()

features = list(credit_df.columns)
features.remove('status')

credit_df[features][0:5]

#Encode the categorical features
encoded_credit_df = pd.get_dummies(credit_df[features], drop_first=True)

y = credit_df.status
X = encoded_credit_df

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = DecisionTreeClassifier(criterion='gini', max_depth=3)

classifier.fit(X_train, y_train)

#Measuring accuracy on test set
y_pred = classifier.predict(X_test)

from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_test, y_pred)

#Finding optimal hyperparameters using GridSearchCV
from sklearn.model_selection import GridSearchCV

params = {'criterion': ['gini', 'entropy'], 
           'max_depth': range(2,10)}

classifier = DecisionTreeClassifier()
clf = GridSearchCV(classifier, param_grid=params, cv=10, scoring="roc_auc")
clf.fit(X_train, y_train)

#Let's look at the best score and best params
print(clf.best_score_)
print(clf.best_params_)



