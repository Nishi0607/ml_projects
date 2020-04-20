# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 02:03:25 2020

@author: NK
"""

import pandas as pd

data = pd.read_csv('bank.csv')

#Upsample (Oversample) the +ve data
from sklearn.utils import resample

data_subscribed = data[data.subscribed=='yes']
data_not_subscribed = data[data.subscribed=='no']

resampled_data_subscribed = resample(data_subscribed, replace=True, n_samples=2000)

from sklearn.utils import shuffle

data_new = pd.concat([resampled_data_subscribed, data_not_subscribed])
#shuffling so that order doesn't impact the modeling
data_new = shuffle(data_new)

features = list(data_new.columns)
features.remove('subscribed')

X = data_new[features]
y = data_new.subscribed
y = y.map(lambda x: 1 if x=='yes' else 0)

categorical_vars = ['job', 'marital', 'education', 'default', 'housing-loan', 'personal-loan']

X_encoded = pd.get_dummies(X, columns=categorical_vars, drop_first=True)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, 
                                                    random_state=42)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, max_depth=10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn import metrics
print(metrics.classification_report(y_test, y_pred))

metrics.roc_auc_score(y_test, y_pred)


