# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:42:16 2020

@author: NK
"""

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn//datasets/SAheart.data')

data.head().iloc[:, 1:8]

count_df = pd.DataFrame(data.chd.value_counts())
#It's an imbalanced dataset
#plotting the counts
count_df.head()
plt.bar(count_df.index, count_df.chd)
plt.xticks(list(count_df.index))
plt.ylabel('Sample count')
plt.show()

#Preprocessing: Encode famhist column
cols_to_encode = ['famhist']
data_encoded = pd.get_dummies(data, columns=cols_to_encode, drop_first=True)

#Upsampling of chd cases which are quite less than no chd
from sklearn.utils import resample, shuffle

chd_df = data_encoded[data_encoded.chd==1]
no_chd_df = data_encoded[data_encoded.chd==0]

chd_df_upsampled = resample(chd_df, n_samples = len(no_chd_df))

data_upsampled = shuffle(pd.concat([no_chd_df, chd_df_upsampled]))

features = list(data_encoded.columns)
features.remove('row.names')
features.remove('chd')

X = data_encoded[features]
y = data_encoded.chd

X_up = data_upsampled[features]
y_up = data_upsampled.chd

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

X_up_train, X_up_test, y_up_train, y_up_test = train_test_split(X_up, y_up, 
                                                                test_size=0.2, 
                                                                random_state=40)
#Building logistic regression using both the datasets
#1. Imbalanced dataset
from sklearn.linear_model import LogisticRegressionCV

logit1 = LogisticRegressionCV(cv=5, scoring="roc_auc")
logit1.fit(X_train, y_train)

logit2 = LogisticRegressionCV(cv=5, scoring="roc_auc")
logit2.fit(X_up_train, y_up_train)

logit1_score = logit1.score(X_test, y_test)
logit2_score = logit2.score(X_up_test, y_up_test)

print('Logit1 roc_auc score: ', logit1_score)
print('Logit2 roc_auc score: ', logit2_score)
#Logit1 roc_auc score:  0.8214624881291549
#Logit2 roc_auc score:  0.7718579234972678

#Build random forest using balanced dataset
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

params = {'n_estimators': [50,100,200,500], 
          'max_depth':[3,5,7,9],
          'max_features':[0.1,0.2,0.3,0.5]}

from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(estimator=rfc, param_grid=params, scoring="roc_auc")
clf.fit(X_up_train, y_up_train)

clf.best_params_
clf.best_score_
#{'max_depth': 9, 'max_features': 0.1, 'n_estimators': 200}

rfc_best = RandomForestClassifier(n_estimators=500, max_depth=9, max_features=0.2)
rfc_best.fit(X_up_train, y_up_train)

rfc_best.feature_importances_
features_imp_df = pd.DataFrame({'feature':X_up.columns, 'importance': rfc_best.feature_importances_})
features_imp_df.sort_values(by='importance', ascending=True, inplace=True)
plt.barh(features_imp_df.feature, features_imp_df.importance,align='center')

features_imp_df.sort_values(by='importance', ascending=False, inplace=True)
features_imp_df['cum_imp'] = features_imp_df['importance'].cumsum()
features_imp_df

features.remove('famhist_Present')
features_95pct = features
#Build a Decision tree using these features
from sklearn.tree import DecisionTreeClassifier

X_dt = data_upsampled[features_95pct]
y_dt = data_upsampled.chd

dtClf = DecisionTreeClassifier(criterion="entropy", max_depth=4)
dtClf.fit(X_dt, y_dt)

