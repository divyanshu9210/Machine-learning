import numpy as np
import matplotlib.pyplot as py
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2]
y = dataset.iloc[:, 2]

"""from sklearn.model_selection import train_test_split
x_train, y_train, x_test, y_test = train_test_split(arrays = x, y, test_size = 0.2, random_state = 123)"""

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x, y)

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree = 4)
x_poly = pf.fit_transform(x);
pf.fit(x_poly, y)

lr2 = LinearRegression()
lr2.fit(x_poly, y)

#visualizing the results as graph(Linear Regression)
py.scatter(x, y, color = 'green')
py.plot(x, lr.predict(x), color = 'red')
py.xlabel('Level')
py.ylabel('Annual Salary')
py.title('Salary Vs Level')
py.show()

#visualizing the results as polynomial regression
py.scatter(x, y, color = 'green')
py.plot(x, lr2.predict(pf.fit_transform(x)), color = 'red')
py.xlabel('Level')
py.ylabel('Annual Salary')
py.title('Salary Vs Level')
py.show()


lr.predict([[6.5]])
lr2.predict(pf.fit_transform([[6.5]]))
#Employee Salary Prediction