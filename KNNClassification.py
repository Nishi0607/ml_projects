# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 01:02:12 2020

@author: NK
"""

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

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

categorical_vars = ['job', 'marital', 'education', 'default', 'housing-loan', 'personal-loan', ]

X_encoded = pd.get_dummies(X, columns=categorical_vars, drop_first=True)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y)

#Default: n_neighbours=5, metric=minkowski
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

import matplotlib.pyplot as plt
from sklearn import metrics

def drawRocCurve(model, X_test, y_test):
    pred_probs = pd.DataFrame(model.predict_proba(X_test))
    test_res_df = pd.DataFrame({'actual': y_test})
    test_res_df=test_res_df.reset_index()
    test_res_df['pred'] = pred_probs.iloc[:, 1:]
    fpr, tpr, thresholds = metrics.roc_curve(test_res_df.actual, test_res_df.pred,
                                                 drop_intermediate=False)
    
    plt.figure(figsize=(16,6))
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1])
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    auc_score = metrics.roc_auc_score(test_res_df.actual, test_res_df.pred)
    return auc_score, fpr, tpr, thresholds

auc_score, fpr, tpr, thresholds = drawRocCurve(knn, X_test, y_test)

print(metrics.classification_report(y_test, knn.predict(X_test)))