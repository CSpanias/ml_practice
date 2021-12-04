# importing required modules
import warnings
warnings.filterwarnings('ignore')
# module for manipulating dataframes (= matrices), series (= single column)
import pandas as pd
# module for manipulating numerical data, arrays (= lists)
import numpy as np
# modules for visualization
import seaborn as sns
import matplotlib.pyplot as plt

# import the required machine learning model
from sklearn.linear_model import LinearRegression
# import a specific function to split our data into training and test data
from sklearn.model_selection import train_test_split
# import matplotlib for visualization
#%matplotlib inline

# loading the dataset and assigning it to the df (DataFrame) variable
filepath = "C:/Users/10inm/Desktop/ML practice/linear_regression_datasets/insurance.csv"
df = pd.read_csv(filepath)

# check how many (rows, columns) this dataset has
# rows = observations (houses)
# columns = features (such as square meters, number of bedrooms, etc.)
print(df.shape)
print(df.head())
df.info(verbose=True)


X = df[['age', 'sex', 'smoker', 'bmi', 'children']]
print(X.shape)

X = pd.get_dummies(data=X, drop_first=True)
print(X.head())
X = X.values
y = df['charges'].values
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 30)

reg = LinearRegression(fit_intercept=True)
reg.fit(X_train, y_train)
score = reg.score(X_test, y_test)
print(score)

predictions = reg.predict(X_test)
sns.regplot(y_test, predictions)
plt.show()

