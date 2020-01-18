# Data analytics for avoiding food wastage

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('FoodData.csv')
# Creating 2 tables one for X which is independent variable and y which is the dependent variable
# For creating X, take all rows and all columns except last one
X = dataset.iloc[:, :-1].values

# While creating y take all the rows and the 3rd column
y = dataset.iloc[:, 4].values

# Encoding categorical data
#We need to split the data for training and testing
#We are training our program to predict the amount of wastage for a given quantity of all 3 types of cuisine

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

# test_size=0.2 means 20% of the data is available for testing and rest 80% is used for training our program
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set

#We use linear regression algorithm for our work
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Predicting the Test set results
y_pred = regressor.predict(X_test)