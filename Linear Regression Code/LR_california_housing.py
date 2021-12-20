import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

import warnings
warnings.filterwarnings('ignore')

# loading the dataset as a pandas DataFrame
california_housing = fetch_california_housing(as_frame=True)

# checking that is a pandas DataFrame
print(california_housing['data'].dtypes)
# checking the Description of the dataset
print(california_housing['DESCR'], "\n")
# checking the target which is named 'MedHouseVal'
print(california_housing['target'].head(), "\n")
# checking the features
print(california_housing['data'].head(), "\n")
# checking the data types and null values
print(california_housing['data'].info(), "\n")

df_X = california_housing.data
df_y = california_housing.target

# convert dataframe to a 2D array
X = df_X[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population',
    'AveOccup']].values
# check that is a 2D array
print(X.shape)
# convert series to a 1D array
y = df_y.values
# check that is a 1D array
print(y.shape)

# split the dataset into training (75%) and testing sets (25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
    random_state=40)

# select the model
model = LinearRegression(fit_intercept=True, normalize=True)
# train the model
model.fit(X_train, y_train)
# predict based on the test data
y_pred = model.predict(X_test)
# evalute model's accuary
score = model.score(X_test, y_test)
print(f"\nModel's accuracy: {round(score, 2)}")




