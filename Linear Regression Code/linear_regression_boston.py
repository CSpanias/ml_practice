# importing required modules

# module for manipulating dataframes (= matrices), series (= single column)
import pandas as pd
# module for manipulating numerical data, arrays (= lists)
import numpy as np
# import the required machine learning model
from sklearn.linear_model import LinearRegression
# import a specific function to split our data into training and test data
from sklearn.model_selection import train_test_split
# import digits dataset that we will work with
from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings('ignore')

# loading digits dataset and assigning it to the boston variable
boston = load_boston()

# check how many (rows, columns) this dataset has
# rows = observations (houses)
# columns = features (such as square meters, number of bedrooms, etc.)
print(boston.data.shape)


"""
splitting the dataset into training (75%) and testing data (25%)

training data = x_train, y_train
x_train = the features (independent variables) that we will train our model with
y_train = the target (dependent variable), the label we want our model 
to be able to predict, examples for our model

testing data = x_test, y_test (new data that our model has never seen before)
"""

# test_size=0.25 puts aside the 25% of our data for later testing
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, 
    test_size=0.25, random_state=0)

# make an instance (a copy) of the model so we can use it later
model = LinearRegression()

"""
train the model on the training data, i.e. the model is learning the 
relationship between features (x_train) and target (y_train)
"""
model.fit(x_train, y_train)

# predict labels for new data (new houses)

# predict for just one observation (house)
print(model.predict(x_test[0].reshape(1,-1)))
# predict for multiple observations (10 houses)
print(model.predict(x_test[0:10]))
# make predictions on entire test data
predictions = model.predict(x_test)
# use score method to get acuracy of the model
score = model.score(x_test, y_test)
print(score)

