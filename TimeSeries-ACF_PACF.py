# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:52:52 2020

@author: NK
"""

import pandas as pd

vimana = pd.read_csv('vimana.csv')
vimana.head()
vimana.info()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#Show autocorrelations upto lag 20
acf_plot = plot_acf(vimana['demand'], lags=20)

#Show partial autocorrelations upto lag 20
pacf_plot = plot_pacf(vimana['demand'], lags=20)

