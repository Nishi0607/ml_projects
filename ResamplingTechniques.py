# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 22:42:50 2020

@author: NK
"""

#Resampling when dataset is imbalanced

import pandas as pd

data = pd.read_csv('bank.csv')

data.head()

data['subscribed'].value_counts()  #4000 not subscribed and 521 subscribed => very imbalanced dataset

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

#Applying logistic regression now
from sklearn.linear_model import LogisticRegression
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

logit = LogisticRegression()
logit.fit(X_train, y_train)

y_pred = logit.predict(X_test)

def draw_conf_mat(actual, predicted):
    conf_mat = confusion_matrix(actual, predicted, [1,0])
    plt.figure(figsize=(16,6))
    sn.heatmap(conf_mat, annot=True, fmt="0.2f", xticklabels=["subscribed", "not subscribed"]
                                                , yticklabels=["subscribed", "not subscribed"])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
draw_conf_mat(y_test, y_pred)

print(classification_report(y_test, y_pred))

#ROC AUC performance analysis
predicted_probs = pd.DataFrame(logit.predict_proba(X_test))
predicted_probs.head()

test_df = pd.DataFrame({'actual':y_test})
test_df = test_df.reset_index()
test_df['pred'] = predicted_probs.iloc[:, 1:2]

from sklearn import metrics
metrics.roc_auc_score(test_df['actual'], test_df['pred'])

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

auc_score, fpr, tpr, thresholds = drawRocCurve(logit, X_test, y_test)
