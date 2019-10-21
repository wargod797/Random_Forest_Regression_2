# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:11:53 2019

@author: sridhar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing a Dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Fitting the Dataset Random Forest

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 30000, random_state = 0)
regressor.fit(X, y)

#Prediction

y_pred = regressor.predict([[6.5]])

#Visualization
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()