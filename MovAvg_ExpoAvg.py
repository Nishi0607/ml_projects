# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:34:21 2020

@author: NK
"""

import pandas as pd

pd.set_option('display.float_format', lambda x: '%.2f' % x)

wsb = pd.read_csv('wsb.csv')

wsb.head()

#Forecast Sale Qty using Moving Average
wsb['mavg_12'] = wsb['Sale Quantity'].rolling(window=12).mean().shift(1)
wsb[['Sale Quantity', 'mavg_12']][12:]

#Plot actual vs Mov Avg values
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10,6))
plt.xlabel('Months')
plt.ylabel('Sale Quantity')
plt.plot(wsb['Sale Quantity'][12:])
plt.plot(wsb['mavg_12'][12:])
plt.show()

#Forecast Accuracy
def get_MAPE(actual, predicted):
    #y_true, y_pred = np.array(actual), np.array(predicted)
    return np.round(np.mean(np.abs((actual-predicted)/actual))*100, 2)

wsb.head()
get_MAPE(wsb['Sale Quantity'][12:].values, wsb['mavg_12'][12:].values)

#RMSE
from sklearn import metrics

np.sqrt(metrics.mean_squared_error(wsb['Sale Quantity'][12:].values, wsb['mavg_12'][12:].values))

##Exponential smoothing
wsb['ewm'] = wsb['Sale Quantity'].ewm(alpha=0.2).mean()

plt.figure(figsize=(10,6))
plt.xlabel('Months')
plt.ylabel('Sale Quantity')
plt.plot(wsb['Sale Quantity'][12:])
plt.plot(wsb['ewm'][12:])
plt.show()

#Get MAPE
get_MAPE(wsb['Sale Quantity'][12:].values, wsb['ewm'][12:].values)


