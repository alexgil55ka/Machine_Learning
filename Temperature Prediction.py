# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:34:37 2019

@author: Alex
"""
# Temperature Prediction

import pandas as pd
from datetime import datetime 
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

def predictTemperature(startDate, endDate, temperature, n):
    # Write your code here
    d1 = datetime.strptime(startDate, "%Y-%m-%d")
    d2 = datetime.strptime(endDate, "%Y-%m-%d")
    days =  abs((d2 - d1).days)

    df = pd.DataFrame({'Hours':  range(0,len(temperature)), 'Temp': temperature})
    X = np.array(df.Hours).reshape((len(df.Hours), 1))
    y = np.array(df.Temp).reshape((len(df.Temp), 1))
    gbr = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.001)
    gbr.fit(X, y.ravel())
    predictions = gbr.predict(np.array(df.Hours).reshape((len(df.Hours), 1)))
    prediction_df = pd.DataFrame({'Temp': temp, 'Prediction':predictions})
    prediction_df.index.name = 'Hours'
    prediction_df['abs_dif'] = abs(prediction_df['Temp'] - prediction_df['Prediction'])
    #prediction_df['result'] = ['correct' for x in prediction_df['abs_dif'] if x < 0.5]
    print(prediction_df)
    
# Test
start = '2014-01-01'
end = '2014-01-01'
temp = [34.38,
 34.36,
 34.74,
 35.26,
 35.23,
 35.29,
 35.64,
 36.02,
 36.1,
 36.98,
 37.01,
 36.75,
 36.01,
 35.66,
 34.72,
 33.9,
 32.62,
 31.51,
 30.73,
 29.5,
 26.94,
 25.47,
 23.84,
 22.55]


predictTemperature(start, end, temp, 1)
    
    

