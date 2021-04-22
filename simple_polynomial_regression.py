import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

# Creating a non-linear relationship data set
x = np.random.normal(0, 1, 70) * 10
y = np.random.normal(-100, 100, 70) + (-x ** 2) * 10

# Plotting dataset
plt.figure(figsize=(10, 5))
plt.scatter(x, y, s=20)
plt.xlabel('input x', fontsize=16)
plt.ylabel('Target', fontsize=16)
plt.show()

# Creating linear Regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x.reshape(-1, 1), y.reshape(-1, 1))
y_pred = lr.predict(x.reshape(-1, 1))

# Plotting linear model with ploy feature data
plt.figure(figsize=(10, 5))
plt.scatter(x, y, s=20)
plt.plot(x, y_pred, color='r')
plt.xlabel('input x', fontsize=16)
plt.ylabel('Target', fontsize=16)
plt.show()

# Printing Linear Regression Error value
print('RMSE for Linear Regression: ', np.sqrt(mean_squared_error(y, y_pred)))

# POLYNOMIAL REGRESSION
# importing polynomial features librery and pipline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Creating pipeline
poly_model = Pipeline([('polynomial', PolynomialFeatures(degree=2)), ('linear', LinearRegression())])
poly_model.fit(x.reshape(-1, 1), y.reshape(-1, 1))
poly_pred = poly_model.predict(x.reshape(-1, 1))

# Visualizing the Polynomial regression
# sorting predicted values with respect to predictor
sorted_zip = sorted(zip(x, poly_pred))
x_poly, poly_pred1 = zip(*sorted_zip)
# plotting predictions
plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=15)
plt.plot(x, y_pred, color='r', label='Linear Regression')
plt.plot(x_poly, poly_pred1, color='g', label='Polynomial Regression')
plt.xlabel('input x', fontsize=16)
plt.ylabel('Target', fontsize=16)
plt.legend()
plt.show()

# Printing Polynomial Regression Error value
print('RMSE for Polynomial Regression=>', np.sqrt(mean_squared_error(y, poly_pred)))
