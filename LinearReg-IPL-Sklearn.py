# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 19:09:24 2020

@author: NK
"""

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

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()

linreg.fit(X_train,y_train)

print(linreg.coef_)
linreg.intercept_

#Store columns and beta coefficients in a data frame and display
coef_df = pd.DataFrame({'feature':X_encoded.columns, 'coef':linreg.coef_})

coef_df = coef_df.sort_values(by='coef', ascending=False)

import matplotlib.pyplot as plt
import seaborn as sn

plt.figure(figsize=(16,6))
sn.barplot(x="coef", y="feature", data=coef_df)
plt.ylabel("Column name")
plt.xlabel("Coefficient value")
