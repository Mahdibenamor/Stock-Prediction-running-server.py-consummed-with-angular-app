# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 22:08:15 2019

@author: Mahdi Ben Amor
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import  pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM


url='C:/Users/Mahdi Ben Amor/Desktop/stock-prediction/price-data.csv';
df = pd.read_csv(url);
#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)
dataset = new_data.values;
train = dataset[0:987,:]
valid = dataset[987:,:]
#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit(dataset)
scaled_data = scaler.transform(dataset)

path="C:/Users/Mahdi Ben Amor/Desktop/stock-prediction/Stock-prediction-model-deploy/scaler.pkl"
file=open(path, 'wb') 
pickle.dump(scaler, file)
file.close()


x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (927,60,1))


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
model.save("C:/Users/Mahdi Ben Amor/Desktop/stock-prediction/Stock-prediction-model-deploy/model.h5")
del model

X_test = [] 
X_test=[153.7000008 , 148.45000044, 148.1500001 , 152.95000106,
152.49999944, 150.69999964, 150.95000103, 149.04999889,
148.65000067, 146.85000086, 146.89999981, 148.29999916,
147.50000048, 147.40000037, 147.50000048, 144.99999989,
144.25000016, 151.60000065, 152.95000106, 155.64999965,
155.99999893, 154.85000098, 158.2999993 , 161.10000023,
163.4000006 , 163.24999932, 163.34999943, 159.5499996 ,
161.30000046, 161.00000012, 155.30000037, 154.19999913,
156.94999889, 156.7500009 , 156.94999889, 157.84999991,
159.40000054, 159.90000111, 155.2499992 , 154.19999913,
155.84999988, 158.25000036, 157.7499998 , 160.90000001,
161.64999974, 163.74999988, 160.0999991 , 157.44999946,
157.3000004 , 156.35000044, 156.29999927, 158.35000047,
156.4999995 , 155.54999954, 154.6999997 , 153.34999928,
152.35000039, 156.39999939, 157.50000063, 160.25000039]
X_test = scaler.transform([X_test])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (1,60,1))
print(X_test)
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)



