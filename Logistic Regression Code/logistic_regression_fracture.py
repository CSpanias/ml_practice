"""
Logistic Regression model with sklearn.

Tried three different models by changing the parameters.

Used KFold to estimate model's mean accuracy.

Used KFold to compare three different models.
"""
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold

import warnings
warnings.filterwarnings('ignore')

# reading the data
df = pd.read_csv("https://www.dropbox.com/s/c6mhgatkotuze8o/bmd.csv?dl=1")

# get a feel of the data
print(df.head())
print(df.shape)

# a dataset made to predict if there is or is not a fracture
print(df.info())
print(df.fracture.unique())

# select features and target
X = df[['age', 'sex', 'weight_kg', 'waiting_time', 'bmd']]
# transform target into boolean
df['fracture_bool'] = df['fracture'] == "fracture"
y = df['fracture_bool']
# transform features into boolean
X = pd.get_dummies(data=X, drop_first=True)
X = X.values
print(X.shape)
y = y.values
print(y.shape)

# split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
                                                    random_state = 40)
# create different classes of the model with different settings
model = LogisticRegression()
model_fit = LogisticRegression(fit_intercept=True)
model_ll = LogisticRegression(solver='liblinear')
# train the model with the training data
model.fit(X_train, y_train)
model_fit.fit(X_train, y_train)
model_ll.fit(X_train, y_train)
# predict on the test data
y_pred = model.predict(X_test)
y_pred_fit = model_fit.predict(X_test)
y_pred_ll = model_ll.predict(X_test)

# Using KFold Cross Validation to evalute model's accuracy.

# creating a function for evaluting the model
def score_model(X, y, model):
    """Calculate the model's mean accuracy."""

    # n_splits = k, shuffle = randomize order of data
    kf = KFold(n_splits = 5, shuffle=True)
    accuracy_scores = [] 
    for train_index, test_index in kf.split(X): 
        X_train, X_test = X[train_index], X[test_index] 
        y_train, y_test = y[train_index], y[test_index] 
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy_scores.append(metrics.accuracy_score(y_test, y_pred))  
    return round(np.mean(accuracy_scores), 2)


# check accuracy score
print("\nModel's mean accuracy with default settings:", score_model(X, y, model))
print("Model with default settings:", round(metrics.accuracy_score(y_test, y_pred), 2))
print("\nModel's mean accuracy with fit intercept:", score_model(X, y,
 model_fit))
print("Model with fit intercept:", round(metrics.accuracy_score(y_test, y_pred_fit), 2))
print("\nModel's mean accuracy with different solver:", score_model(X,
 y, model_ll))
print("Model with different solver:", round(metrics.accuracy_score(y_test, y_pred_ll),
 2), "\n")

"""
Use K-Fold Cross Validation to compare 3 different models:

1. A Logistic Regression with all features
2. A Logistic Regression with age, weight, height, sex.
3. A Logistic Regression with age, weight.
"""
X1 = df[['age', 'sex', 'weight_kg', 'waiting_time', 'bmd', 'medication',
    'height_cm']]
X2 = df[['age', 'weight_kg', 'height_cm', 'sex']]
X3 = df[['age', 'weight_kg']]

# transform features and target into boolean
X1 = pd.get_dummies(data=X1, drop_first=True)
X1 = X1.values
X2 = pd.get_dummies(data=X2, drop_first=True)
X2 = X2.values
X3 = pd.get_dummies(data=X3, drop_first=True)
X3 = X3.values
print(X.shape)
df['fracture_bool'] = df['fracture'] == "fracture"
y = df['fracture_bool'].values
print(y.shape,"\n")

# k, randomize order of data
kf = KFold(n_splits = 5, shuffle=True)

def score_model(X, y, kf): 
    accuracy_scores = [] 
    precision_scores = [] 
    recall_scores = [] 
    f1_scores = [] 
    for train_index, test_index in kf.split(X): 
        X_train, X_test = X[train_index], X[test_index] 
        y_train, y_test = y[train_index], y[test_index] 
        model = LogisticRegression(solver='liblinear') 
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test) 
        accuracy_scores.append(metrics.accuracy_score(y_test, y_pred)) 
        precision_scores.append(metrics.precision_score(y_test, y_pred)) 
        recall_scores.append(metrics.recall_score(y_test, y_pred)) 
        f1_scores.append(metrics.f1_score(y_test, y_pred)) 
    print("accuracy:", round(np.mean(accuracy_scores), 2)) 
    print("precision:", round(np.mean(precision_scores), 2)) 
    print("recall:", round(np.mean(recall_scores), 2)) 
    print("f1 score:", round(np.mean(f1_scores), 2))
    return ""


print("Logistic Regression with all features:")
print(score_model(X1, y, kf))
print("\nLogistic Regression with age, weight, height and sex:")
print(score_model(X2, y, kf))
print("\nLogistic Regression with age and weight:")
print(score_model(X3, y, kf))
