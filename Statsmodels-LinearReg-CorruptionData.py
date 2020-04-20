# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 14:06:23 2020

@author: NK
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

data = pd.read_csv('country.csv')

data.info()

X = sm.add_constant(data['Gini_Index'])
y = data['Corruption_Index']

model = sm.OLS(y,X).fit()

print(model.params)
#Corruption_Index = -1.295 * Gini_Index  + 106.695

print(model.summary2())

#95% confidence interval for b1: b1 +/- t-crit * Std Error

summary = model.summary2()

conf_interval = model.conf_int(alpha=0.05).ix['Gini_Index']

