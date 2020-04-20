# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 14:38:57 2020

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

list(encoded_credit_df.columns)

credit_df['credit_history'].unique()

encoded_credit_df[['credit_history_A31', 'credit_history_A32', 'credit_history_A33', 'credit_history_A34']].head()

#Building the model
y = credit_df.status
X = sm.add_constant(encoded_credit_df)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

logit_model = sm.Logit(y_train, X_train).fit()

print(logit_model.summary2())

#Get features which are significant at significance level of 0.05
def getSignificantVars(model):
    p_vals_df = pd.DataFrame(model.pvalues)
    p_vals_df['vars'] = p_vals_df.index
    p_vals_df.columns = ['pvals', 'vars']
    return list(p_vals_df[p_vals_df.pvals <= 0.05]['vars'])

significant_columns = getSignificantVars(logit_model)
final_logit_model = sm.Logit(y_train, sm.add_constant(X_train[significant_columns])).fit()

final_logit_model.summary2()

#Predict on test data
y_pred = final_logit_model.predict(sm.add_constant(X_test[significant_columns]))

y_pred_df = pd.DataFrame({'actual':y_test, 'pred_prob':y_pred})

y_pred_df['predicted'] = y_pred_df.pred_prob.map(lambda x: 1 if x>0.5 else 0)
y_pred_df.head(12)

y_pred_df.sample(12,random_state=42)

#Create confusion matrix
from sklearn import metrics
import seaborn as sn

def draw_conf_matrix(actual, predicted):
    conf_mat = metrics.confusion_matrix(actual, predicted, labels=[1,0])
    sn.heatmap(conf_mat, annot=True, fmt=".2f", 
               xticklabels=["Bad Credit", "Good Credit"],
               yticklabels=["Bad Credit", "Good Credit"])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.figure(figsize=(26,15))
    plt.show()
    
draw_conf_matrix(y_pred_df.actual, y_pred_df.predicted)

print(metrics.classification_report(y_pred_df.actual, y_pred_df.predicted))

def draw_roc(actual,probs):
    fpr,tpr,threshold = metrics.roc_curve(actual, probs, drop_intermediate=False)
    auc_score = metrics.roc_auc_score(actual, probs)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label = "ROC Curve (area=%0.2f)" % auc_score)
    plt.plot([0,1], [0,1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc="lower right")
    plt.show()
    return fpr,tpr,threshold

fpr,tpr,threshold = draw_roc(y_pred_df['actual'], y_pred_df['pred_prob'])

#Improving the model by choosing the optimal classification cut-off
#Youden's index: Need to choose p for which tpr+tnr-1 is maximized, i.e. tpr - fpr is maximized
tpr_fpr_df = pd.DataFrame({'tpr':tpr, 'fpr':fpr, 'thresholds':threshold})
tpr_fpr_df['diff'] = tpr_fpr_df['tpr'] - tpr_fpr_df['fpr']
tpr_fpr_df.sort_values('diff', ascending=False)[0:5]

#Above gives optimal thresold of 0.265
y_pred_df['predicted_new'] = y_pred_df['pred_prob'].map(lambda x: 1 if x>0.265 else 0)

draw_conf_matrix(y_pred_df.actual, y_pred_df.predicted_new)

print(metrics.classification_report(y_pred_df.actual, y_pred_df.predicted_new))
#print(metrics.classification_report(y_pred_df.actual, y_pred_df.predicted))

metrics.roc_auc_score(y_pred_df.actual, y_pred_df.predicted_new)
