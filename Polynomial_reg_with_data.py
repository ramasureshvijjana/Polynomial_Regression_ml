# importing Required libraries
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# DATA PREPROCESSING
# Data loading
data = pd.read_csv("Fish.csv")
x_y_values = data.loc[:, ['Width', 'Weight']]
print('Required data:\n', x_y_values)
# Checking null values in data
print('\nChecking for null values:\n', x_y_values.isnull().sum())

# Feature scaling------Here no need of feature scaling
# Because No change in accuracy after apply the feature scaling
# from sklearn import preprocessing
# std_scaler = preprocessing.StandardScaler()
# x_y_values = std_scaler.fit_transform(x_y_values)
# x_y_values = pd.DataFrame(x_y_values, columns=['Width', 'Weight'])
# print(x_y_values)

# Selecting x, y values from data
x = x_y_values.loc[:, 'Width'].values.reshape(-1, 1) # Input
y = x_y_values.loc[:, 'Weight'].values.reshape(-1, 1) # Target

# Splitting data for train and test purpose
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=.3,
                                                    random_state=10)
print('\nSize of x_train, x_test, y_train, y_test respectively:',
      x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# POLYNOMIAL REGRESSION
poly_model = Pipeline([('polynomial', PolynomialFeatures(degree=4)), ('linear', LinearRegression())])
poly_model = poly_model.fit(x_train, y_train)
pred = poly_model.predict(x_test)

# Finding error and accuracy
print('\nREMS for Polynomial Regression:', np.sqrt(mean_squared_error(y_test, pred)))
print(' \nAccuracy: ', metrics.explained_variance_score(y_test, pred))

# zip the values for graph plotting
sorted_zip = sorted(zip(x_test, pred))
x_poly, poly_pred = zip(*sorted_zip)

# Plotting graph
plt.xlabel('width')
plt.ylabel('weight')
plt.scatter(x_train, y_train)
plt.plot(x_poly, poly_pred, c='r')
plt.show()
