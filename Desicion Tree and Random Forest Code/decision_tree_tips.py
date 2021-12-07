# import libraries
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

# import dataset
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
print(df.head())

# delete smoker variable
del df['smoker']

# convert non-numeric data using one-hot encoding
df = pd.get_dummies(df, columns=['time', 'day', 'sex'])

# assign X and y variables
X = df.drop('tip', axis=1)
y = df['tip']

# split test/train data and shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    shuffle=True)

# assign algorithm
model = DecisionTreeRegressor()

# link algorithm to X and y variables
model.fit(X_train, y_train)

# check prediction error for training and test data using MAE
y_pred = model.predict(X_test)
mae_train = mean_absolute_error(y_train, model.predict(X_train))
print("Training Set Mean Absolute Error: %.2f" % mae_train)

mae_test = mean_absolute_error(y_test, y_pred)
print("Test Set Mean Absolute Error: %.2f" % mae_test)

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