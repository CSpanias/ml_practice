import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# import dataset
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv")
# drop rows containing missing values
df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
# convert non-numeric data using one-hot encoding
df = pd.get_dummies(df, columns=['sex', 'island'])
# standardize the independent variables using StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('species', axis=1))
scaled_df = scaler.transform(df.drop('species', axis=1))
# assign X and y variables
X = scaled_df
y = df['species']
# split data into training and test data
X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.3,
                                                     shuffle=True)

# assign algorithm
model = SVC()

# link algorithm to X and y variables
model.fit(X_train, y_train)

# run algorithm on test data to make predictions
y_pred = model.predict(X_test)

# evaluate predictions
print(confusion_matrix(y_test, y_pred))
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