#import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# import dataset
df = pd.read_csv(r"https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv")
print(df.head())

# delete sex variable (avoid binary variable for kNN)
del df['sex']

# check dataset (datatypes, size, null values)
print(df.info())
# check for missing values
print(df.isna().sum())
# drop drows containing missing values
df.dropna(axis=0, how='any',thresh=None,subset=None,inplace=True)

# convert non-numeric data using one-hot encoding
df = pd.get_dummies(df, columns=['island'])

# Standardize the features using StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop("species", axis=1))
scaled_df = scaler.transform(df.drop('species', axis=1))

# assign X and y variables
X = scaled_df
y = df['species']

# split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
 shuffle=True)

# assign algorithm
model = KNeighborsClassifier(n_neighbors=5)

# Link algorithm to X and y variables
model.fit(X_train, y_train)

# run algorithm on test data
y_pred = model.predict(X_test)

# evalute predictions
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Data point to predict
# Data point to predict
penguin = [
    39, #bill_length_mm
    18.5, #bill_depth_mm
    180, #flipper_length_mm 
    3750, #body_mass_g
    0, #island_Biscoe    
    0, #island_Dream
    1, #island_Torgersen    
]

# Make prediction
new_penguin = model.predict([penguin])
print(new_penguin)