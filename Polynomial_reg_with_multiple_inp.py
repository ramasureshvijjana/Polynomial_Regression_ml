# importing Required libraries
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("Fish.csv")
print('Required data:\n', data)
print('\nChecking for null values:\n', data.isnull().sum())

leb_enc = LabelEncoder()
data['Species'] = leb_enc.fit_transform(data['Species'])
print('Required data:\n', data)

x = data.loc[:, ['Species', 'Length1', 'Length2', 'Length3', 'Height', 'Width']]
y = data.loc[:, 'Species']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=10)
print('\nSize of x_train, x_test, y_train, y_test respectively:', x_train.shape, x_test.shape, y_train.shape, y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

poly_model = Pipeline([('polynomial', PolynomialFeatures(degree=4)), ('linear', LinearRegression())])
poly_model = poly_model.fit(x_train, y_train)
pred = poly_model.predict(x_test)

print('\nREMS for Polynomial Regression:', np.sqrt(mean_squared_error(y_test, pred)))
print(' \nAccuracy: ', metrics.explained_variance_score(y_test, pred))

sorted_zip = sorted(zip(x_test, pred))
x_poly, poly_pred = zip(*sorted_zip)

