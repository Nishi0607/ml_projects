# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 21:27:12 2019

@author: NK
"""

#Part 1 - Data preprocessing (70% of the task? :))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import the training set
dataset_train = pd.read_csv('AMZN_Px-Trainset.csv')
training_set = dataset_train.iloc[:, 4:5].values

#Feature Scaling
#RNN: Normalization is recommended if Sigmoid is the activation function of the output payer
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# Save time series for 90 previous days and 1 output (i.e. stock px at time T+1)
X_train = []
y_train = []

for i in range(90,len(training_set_scaled)):
    X_train.append(training_set_scaled[i-90:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Part 2 - Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

#Add LSTM layer and Dropout regularization (to avoid overfitting)
regressor.add(LSTM(units = 60, return_sequences=True, input_shape=(X_train.shape[1], 1)))
#Units: no. of neurons in the LSTM layer - needs to be high to capture the up/down trend in the prices precisely
#return_sequences: Needs to be true if we want to add more LSTM layers
#input_shape: Shape of each training data
regressor.add(Dropout(0.25))
#20% of neurons of the LSTM layer will be ignored during training, i.e. during the forward and backward propagation passes

#Add 3 more LSTM layers with dropout regularization
#(input_shape not needed as lstm knows there are 50 neurons in the prev layer)
regressor.add(LSTM(units = 60, return_sequences=True))
regressor.add(Dropout(0.25))

regressor.add(LSTM(units = 60, return_sequences=True))
regressor.add(Dropout(0.25))

regressor.add(LSTM(units = 60))
regressor.add(Dropout(0.25))

#Add the output layer (not a LSTM layer, but a fully connected layer)
regressor.add(Dense(units=1))

#Compiling the RNN
#Usually RMPProp/Adam optimizers are more relevant
#Mean Square Error makes more sense - error can be measured better using mean of sq diff between actual and predicted values
regressor.compile(optimizer = 'Adam', loss='mean_squared_error' )

#Fitting the RNN to the training set
#Epochs to be selected such that there is a convergence of the error upon running as many times
regressor.fit(x=X_train, y=y_train, epochs=100, batch_size=32)
#Observe the convergence of the Loss value across the epochs as they proceed
#Should not try to decrease the loss as much as possible, as that can lead to overfitting
#Epochs 25:Loss converges nicely from .014 -> .0015
#Epochs 50:Loss converges nicely from .0086 -> .001
#Epochs 100:Loss converges nicely from .0092 -> .00081

#Part 3 - Making predictions and visualizing results

#get the real closing stock prices of 30 days starting Jul01, 2019
dataset_test = pd.read_csv('AMZN_Px-Testset.csv')
real_stock_price = dataset_test.iloc[:, 4:5].values

#Get the predicted stock price of Jul/Aug 2017 (use 90 previous stock prices for predicting 30 future close prices)
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test) - 90 :].values
inputs =inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(90,len(inputs)):
    X_test.append(inputs[i-90:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
#Inverse scale the predict stock prices
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualizing the results
plt.plot(real_stock_price, color='blue', label='Actual Amzn Close Px')
plt.plot(predicted_stock_price, color='red', label='Pred Amzn Close Px')
plt.title('Amzn Close Px')
plt.xlabel('Day')
plt.ylabel('Amzn ClosePx')

#Model predictions follow the normal trend quite well, though not the interim spikes 


