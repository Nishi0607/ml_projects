# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 19:25:27 2020

@author: NK
"""

from statsmodels.tsa.arima_model import ARIMA

import pandas as pd
import numpy as np

vimana = pd.read_csv('vimana.csv')

#ARIMA model of order (p,d,q) - we set d and q as 0 to get an AR model basically
#1. AR model of order 1
arima = ARIMA(vimana.demand[0:30].astype(np.float64).as_matrix(),order = (1,0,0))

ar_model = arima.fit()

ar_model.summary2()

#Forecasting using the model (ARIMA model in-sample/out-of-sample prediction)
forecast_31_37 = ar_model.predict(30, 36)
forecast_31_37

#Measure the accuracy using MAPE
def get_mape(actual, predicted):
    return np.round(np.mean(np.abs((actual-predicted)/actual))*100, 2)

print(get_mape(vimana.demand[30:], forecast_31_37))

#2. Forecasting using MA model
arima_2 = ARIMA(vimana.demand[0:30].astype(np.float64).as_matrix(), order=(0,0,1))
vimana.head()
ma_model = arima_2.fit()

ma_model.summary2()
#p-value for moving average with lag 1 is < 0.05, hence statistically significant

#Forecast the next 6 periods
forecast_ma_31_37 = ma_model.predict(30,36)
forecast_ma_31_37

#Measure accuracy of the forecast
get_mape(vimana.demand[30:], forecast_ma_31_37)

#3. ARMA Model
arima_3 = ARIMA(vimana.demand[0:30].astype(np.float64).as_matrix(), order=(1,0,1))

arma_model = arima_3.fit()
arma_model.summary2()
#We see that MA with lag 1 is not significant here

#Measure accuracy
forecast_arma_31_37 = arma_model.predict(30,36)
get_mape(vimana.demand[30:], forecast_arma_31_37)

#4. ARIMA model



