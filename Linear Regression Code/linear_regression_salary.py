""" 
Linear Regression using sklearn.

Used KFold Cross Validation to find the model's mean accuracy.
"""

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold

import warnings
warnings.filterwarnings('ignore')

# read the dataset from a cvs file
filepath = "C:\\Users\\10inm\\Desktop\\ml_practice\\linear_regression_datasets\\Salary.csv"
df = pd.read_csv(filepath)

# get a sense of the dataset by seeing the 1st 5 rows
print(df.head(), "\n")
# check total rows, columns
print(df.shape, "\n")
# check for data types and missing values
print(df.info(), "\n")

"""
Build a Linear Regression model to predict salary 
based on years of experience.
"""

# convert the data from Series to arrays and select features and target
X = df[['YearsExperience']].values
# check that is successfuly converted into a 2D array
print(X.shape)
y = df.Salary
# check the is successfult converted into a 1D array
print(y.shape)

# split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
    random_state=10)
# select the model
model = LinearRegression(fit_intercept=True)
# train the model on the training data
model.fit(X_train, y_train)
# predict on the test data
y_pred = model.predict(X_test)
# evaluate the model
score = round(model.score(X_test, y_test), 2)

"""
Use KFold Cross Validation to evalute model's accuracy.
"""

# n_splits = k, shuffle = randomize order of data
kf = KFold(n_splits = 5, shuffle=True)

# creating a function for evaluting the model
def score_model(X, y, kf):
    """A function to find the model's mean accuracy."""

    accuracy_scores = [] 
    for train_index, test_index in kf.split(X): 
        X_train, X_test = X[train_index], X[test_index] 
        y_train, y_test = y[train_index], y[test_index] 
        model = LinearRegression() 
        model.fit(X_train, y_train) 
        accuracy_scores.append(model.score(X_test, y_test))  
    return round(np.mean(accuracy_scores), 2)

# Model's mean accuracy score will change slightly on every run
print("\nModel's mean accuracy score:", score_model(X, y, kf))
print("Model's actual accuracy score:", score)