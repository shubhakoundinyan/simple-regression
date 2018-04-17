#	coding: utf-8
#	@author: Shubha Koundinyan

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Divide the data into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 1/3, random_state = 0)

# Fit the parameters to the Linear Regressor class
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Testing the model built on the Test Set
y_pred = regressor.predict(X_test)

# Visualizing the data for the Test and Train set
plt.scatter(X_train, y_train, color="red", label="Real Values")
plt.plot(X_train, regressor.predict(X_train), color = "blue", label="Predicted")
plt.title("Experience V/S Salary(Train Set)")
plt.xlabel("Experience(years)")
plt.ylabel("Salary(USD)")
plt.legend(loc = "upper-left")
plt.show()

# Visualizing the test set
plt.scatter(X_test, y_test, color = "Red", label="Real")
plt.plot(X_train, regressor.predict(X_train), color = "Blue", label="Predicted")
plt.title("Experience V/S Salary (Test-Set)")
plt.xlabel("Experience(years)")
plt.ylabel("Salary(USD)")
plt.legend(loc = "upper right")
plt.show()