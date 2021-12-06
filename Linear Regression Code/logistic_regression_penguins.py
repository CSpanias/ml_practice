"""
Logistic Regression using sklearn.

Guided tutorial/exercise from the book Machine Learning for Absolute Beginners.

"""
# import libraries
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')
pd.options.display.width = 0

# import dataset
df = sns.load_dataset("penguins")
print("\nThe first five rows of the dataset:")
print(df.head())
print("\n General information about our dataset such as data types, "
      "number of rows and columns, etc.")
print(df.info())

# check for missing values
print("\nMissing values per column:\n")
print(df.isna().sum())

# check how many classes we have for our target variable
print("\nHow many different classes, i.e. penguin species, there are in our"
      "dataset")
print(df.species.unique())

# Drop rows containing missing values (NaNs)
df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
# check for missing values
print("\nMissing values per column after we dropped them:\n")
print(df.isna().sum())

# convert non-numeric data using one-hot encoding
df = pd.get_dummies(df, columns=['sex', 'island'])

# assign X and y variables
X = df.drop('species', axis=1)
y = df['species']

# split data into test/train set (70/30 split) and shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    shuffle=True)

# assign algorithm
model = LogisticRegression()

# link algorithm to X and y variables
model.fit(X_train, y_train)

# run algorithm on test data to make predictions
y_pred = model.predict(X_test)

# evaluate predictions
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Data point to predict
penguin = [
	39, #bill_length_mm
	18.5, #bill_depth_mm
	180, #flipper_length_mm
	3750, #body_mass_g
	0, #island_Biscoe
	0, #island_Dream
	1, #island_Torgersen
	1, #sex_Male
	0, #sex_Female
]

# Make prediction
new_penguin = model.predict([penguin])
print(new_penguin)