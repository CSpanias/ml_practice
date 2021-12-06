"""
Linear Regression using sklearn.

Guided tutorial/exercise from the book Machine Learning for Absolute Beginners.

"""
# import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import warnings
warnings.filterwarnings('ignore')
pd.options.display.width = 0

# import dataset
df = sns.load_dataset("tips")
# had low correlation with target during EDA
del df['smoker']

# get a feel of the data
print(df.head())
print(df.info)

# exploratory data analysis (EDA): correlation and heatmap
"""
Remove the '#' to run this code. 
"""
df_corr = df.corr()
sns.heatmap(df_corr, annot=True,cmap='coolwarm')
plt.show()

# EDA: pairplot
sns.pairplot(df)
plt.show()

# convert non-numeric data using one-hot encoding
df = pd.get_dummies(df, columns=['time', 'day', 'sex'])
print(df.head())
print(df.shape)

# EDA (2): correlation
"""
As now we have 13 instead of 3 independent variables it is hard to visualize
them using graphs.

That is why I commented ('#') the above EDA, and we will use just correlation.
"""
df_corr = df.corr()
print(df.corr())

# Assign X and y variables

# keep every column, but the ones specified
X = df.drop('tip', axis=1)
y = df['tip']

# Split data into test/train set (70/30 split) and shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
    shuffle=True)

# Assign algorithm
model = LinearRegression()

# Link algorithm to X and y variables
model.fit(X_train, y_train)

# Find y-intercept
print(model.intercept_)

# Find x coefficients
print(model.coef_)

# check how accurate the model is
mae_train = mean_absolute_error(y_train, model.predict(X_train))
# round to 2 decimal places using placeholder
print("\nTraining Set Mean Absolute Error: %.2f" % mae_train)

mae_test = mean_absolute_error(y_test, model.predict(X_test))
print("Test Set Mean Absolute Error: %.2f\n" % mae_test)

# Data point to predict
jamie = [
    40, #total_bill
    2, #size
    1, #time_dinner
    0, #time_lunch
    1, #day_fri
    0, #day_sat
    0, #day_sun
    0, #day_thur
    1, #sex_female
    0, #sex_male
]

# Make prediction
jamie = model.predict([jamie])
print(jamie)
