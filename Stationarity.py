# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:56:37 2020

@author: NK
"""

import pandas as pd

store = pd.read_excel('store.xls')

from statsmodels.graphics.tsaplots import plot_acf
acf_plot = plot_acf(store.demand, lags=20)

#The slow decline of auto-correlations for different lags in the ACF plot suggests possible non-staionarity
#Let's check with Dicky-Fuller test

from statsmodels.tsa.stattools import adfuller

def adfuller_test(tseries):
    adfuller_res = adfuller(tseries, autolag=None)  #None => max lags are used
    adfuller_output = pd.Series(adfuller_res[0:4], 
                                index=['Test statistic',
                                       'p-value',
                                       'Lags Used',
                                       'No. of observations used'])
    print(adfuller_output)

adfuller_test(store.demand)

len(store)

    