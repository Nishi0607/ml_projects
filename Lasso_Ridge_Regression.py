# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:44:48 2020

@author: NK
"""

#LASSO And Ridge Regularization to prevent over-fitting

import numpy as np
import pandas as pd

data = pd.read_csv('IPL IMB381IPL2013.csv')

data.info()

features = list(data.columns)
features.remove('Sl.NO.')
features.remove('SOLD PRICE')
features.remove('PLAYER NAME')

X = data[features]
y = data['SOLD PRICE']

X.head()

categorical_vars = ['AGE', 'COUNTRY', 'TEAM', 'PLAYING ROLE']

X_encoded = pd.get_dummies(X, columns=categorical_vars, drop_first=True)

#Scale the features
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_encoded_scaled = sc.fit_transform(X_encoded)
y_scaled = (y - y.mean()) / y.std()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_encoded_scaled, y_scaled, test_size=0.2, random_state=42)

from sklearn.linear_model import Ridge
from sklearn import metrics
ridge = Ridge(alpha=1, max_iter=500)
ridge.fit(X_train, y_train)

def getTrainTestRMSE(model):
    y_train_pred = model.predict(X_train)
    rmse_train = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
    y_test_pred = model.predict(X_test)
    rmse_test = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    return rmse_train, rmse_test

rmse_train, rmse_test = getTrainTestRMSE(ridge)

print("train:", rmse_train, "test:", rmse_test)

#Try ridge with regularization strength alpha = 2
ridge_2 = Ridge(alpha = 2, max_iter = 1000)
ridge_2.fit(X_train, y_train)
rmse_train_2, rmse_test_2 = getTrainTestRMSE(ridge_2)
print("train:", rmse_train_2, "test:", rmse_test_2)

#LASSO Regularization
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=.01, max_iter=500)
lasso.fit(X_train, y_train)
rmse_train_3,rmse_test_3 = getTrainTestRMSE(lasso)
print("train:", rmse_train_3, "test:", rmse_test_3)

#Lasso would have reduced some of the coefficient values to 0. Let's check
lasso_coef_df = pd.DataFrame({'column': X_encoded.columns, 'coef': lasso.coef_})
lasso_coef_df = lasso_coef_df.sort_values(by='coef', ascending=True)
#Get columns for which the coefficient has been forced to 0
lasso_coef_df[lasso_coef_df.coef==0]

#ElasticNet Regression (combination of L1 and L2 penalty)
from sklearn.linear_model import ElasticNet
#gamma(L1) = 0.01, sigma(L2) = 1
elasticnet = ElasticNet(alpha=1.01, l1_ratio=.0099, max_iter = 500)
elasticnet.fit(X_train, y_train)
rmse_train_enet, rmse_test_enet = getTrainTestRMSE(elasticnet)

print("train:", rmse_train_enet, "test:", rmse_test_enet)

#Gives rmse_train of 0.697 and rmse_test of 0.701, which implies applying both the regularizations did help
#improve the model performance