# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 20:58:56 2020

@author: NK
"""

import statsmodels.api as sm
import numpy as np
import pandas as pd

salary_df = pd.read_csv('MBA Salary.csv')
salary_df.head()

salary_df.info()

type(salary_df['Salary'])

X = sm.add_constant(salary_df['Percentage in Grade 10'])

X.head(10)

y = salary_df['Salary']

#Split the data into training/test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 50)

salary_lm = sm.OLS(y_train, X_train).fit()

print(salary_lm.params)

salary_lm.summary2()

#Residuals analysis: If the cumulative distribution of the residuals is closely around the cd of normal (45-deg line)
#we can conclude tht the residuals are normally distributed
residuals = salary_lm.resid

import matplotlib.pyplot as plt
probplot = sm.ProbPlot(residuals)
plt.figure(figsize=(16,6))
probplot.ppplot(line='45')
plt.title('P-P plot of regression residuals')
plt.show()

#Test of homoscedasticity (use residual plot of standardized residual vs standardized predicted values)
#Heteroscedasticity => funnel-type shape in the residual plot
def getStandardizedValues( vals ):
    return (vals - vals.mean()) / vals.std()

plt.scatter(getStandardizedValues(salary_lm.fittedvalues), getStandardizedValues(salary_lm.resid))
plt.title('Scatter plot of std predictions vs std residuals')
plt.ylabel('Standardized residuals')
plt.xlabel('Standardized predicted values')
plt.show()

#Model diagnostics:
#z-score > 3 => outliers in data
from scipy.stats import zscore
zscores = zscore(salary_df['Salary'])
plt.bar(zscores, height=zscores)

#Prediction
y_pred = salary_lm.predict(X_test)

#Measure accuracy of prediction: r-squared and root mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error
r2_score(y_test, y_pred)

import numpy as np
#mean_squared_error: smaller the better
np.sqrt(mean_squared_error(y_test, y_pred))

#Calculating prediction intervals
from statsmodels.sandbox.regression.predstd import wls_prediction_std
#predict low/high interval values for y:
pred_std_error, pred_y_low, pred_y_high = wls_prediction_std(salary_lm, X_test, alpha=0.1)

#store predicted values and intervals together
pred_test_df = pd.DataFrame({'Grade10_Perc': X_test['Percentage in Grade 10'],
                            'pred_test_y': y_pred,
                            'pred_test_y_low': pred_y_low,
                            'pred_test_y_high': pred_y_high,
                            })

pred_test_df[0:5]

#Multi-collinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
