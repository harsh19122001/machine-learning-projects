# Backend code for Simple Linear Regression

# Simple Linear Regression Model for Salary Prediction  
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import variation

import scipy.stats as stats

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

dataset = pd.read_csv(r"D:\python\machine_learning\1.Regression\Salary_Data.csv")

dataset.head()

# Divide the variable into dependent & independent variable.
x = dataset.iloc[:, :-1] # x is independent variable
y = dataset.iloc[:,-1] # y is dependent variable


# split the dataset into training set and testing sets(80% train and 20% test)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =  train_test_split(x,y, test_size = 0.2, random_state = 0)

# Train the model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict the test set result
y_pred = regressor.predict(x_test)

# Predict the full dataset
y_full_pred = regressor.predict(x)

# compare actual vs predicted
comparsion = pd.DataFrame({'Actual': y_test, 'Prediction' : y_pred})
print(comparsion)


# visualize 
plt.scatter(x, y, color = 'red', label = 'Actual Salary') # Real Salary data
plt.plot(x, regressor.predict(x), color = 'blue') # regression line
plt.title('Actual vs Regerssion line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# getting the slope and intercept
m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

# predict salary for 12 and 20 years of experience
y_12 = m_slope * 12 + c_intercept
print(y_12)

y_20 = m_slope * 20 + c_intercept
print(y_20)

# model evaluation
bias_score = regressor.score(x_train, y_train)
print("bias_score",bias_score)

# variance
variance_score = regressor.score(x_test,y_test)
print("variance_score",variance_score)

# stats integration to ml

# Mean
dataset.mean()

# Salary column mean
dataset['Salary'].mean()

# Median
dataset.median()

# Salary column median
dataset['Salary'].median()

# Mode
dataset.mode()

# Salary column mode
dataset['Salary'].mode()

# Variance
dataset.var()

# Salary column variance
dataset['Salary'].var()

# Standard Deviation
dataset.std()

# Salary column standard deviation
dataset['Salary'].std()

# Cofficient of variation
variation(dataset.values)

# Salary column cofficient of variation
variation(dataset['Salary'])

# Correlation
dataset.corr()

# Salary column correlation with YearsExperience
dataset['Salary'].corr(dataset['YearsExperience'])

# Skewness
dataset.skew()

# Salary column skewness
dataset['Salary'].skew()

# standard error of mean
dataset.sem()

# Salary column standard error of mean
dataset['Salary'].sem() # this will give us standard error of that particular column

# Z-score

# for calculating Z- score we have to import a library first

dataset.apply(stats.zscore) # this will give Z- score of entire dataframe

stats.zscore(dataset['Salary']) # this will give us Z- score of that particular column

# Degree of Freedom

a = dataset.shape[0] # this will give us no. of rows
b = dataset.shape[1] # this will give us no. of columns

degree_of_freedom = a - b
print(degree_of_freedom) # this will give us degree of freedom for entire dataset

# inova

# ssr
y_mean = np.mean(y)
SSR = np.sum((y_full_pred - y_mean)**2)
print(SSR)

# sse
SSE = np.sum((y - y_full_pred) ** 2)
print(SSE)

# sst
y_mean = np.mean(y)
SST = np.sum((y - y_mean)**2)
print(SST)

#r2
r_square = 1 - SSR/SST
print(r_square)

bias = regressor.score(x_train, y_train)
print(bias)

variance = regressor.score(x_test, y_test)
print(variance)

import pickle
# Save the trained model to disk
filename = 'linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as linear_regression_model.plk")

import os
os.getcwd()
print(os.getcwd())