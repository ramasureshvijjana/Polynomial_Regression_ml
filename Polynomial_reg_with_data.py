# importing Required libraries
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv("Fish.csv")
x_y_values = data.loc[:,['Width', 'Weight']]
print('Required data:\n',x_y_values)
print('\nChecking for null values:\n', x_y_values.isnull().sum())

# from sklearn import preprocessing
# std_scaler = preprocessing.StandardScaler()
# x_y_values = std_scaler.fit_transform(x_y_values)
# x_y_values = pd.DataFrame(x_y_values, columns=['Width', 'Weight'])
# print(x_y_values)

x = x_y_values.loc[:, 'Width'].values.reshape(-1, 1)
y = x_y_values.loc[:, 'Weight'].values.reshape(-1, 1)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=10)
print('\nSize of x_train, x_test, y_train, y_test respectively:',x_train.shape, x_test.shape, y_train.shape, y_test.shape)

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

plt.xlabel('width')
plt.ylabel('weight')
plt.scatter(x_train, y_train)
plt.plot(x_poly, poly_pred, c='r')
plt.show()
