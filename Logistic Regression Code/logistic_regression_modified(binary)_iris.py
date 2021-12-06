"""
Binary Logistic Regression using sklearn.

Built four different models.

Used multiple evaluation metrics.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

import warnings
warnings.filterwarnings('ignore')

# import the dataset file
filepath = "C:\\Users\\10inm\\Desktop\\ml_practice\\logistic_regression_datasets\\modifiedIris2Classes.csv"
df = pd.read_csv(filepath)

# check the dataset
print(df.shape)
print(df.head())

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['petal length (cm)']], 
df['target'], test_size=0.25, random_state=0)

# Build a model with the default settings
clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
score = clf.score(X_test, y_test)
print(f"Default model:\nAccuracy: {round(score,2)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1: {round(f1_score(y_test, y_pred),2)}\n")
print(classification_report(y_test, clf.predict(X_test)), "\n")
print(confusion_matrix(y_test, clf.predict(X_test)), "\n")

# Build a model with normalisation
# everything in cm, thus, won't change much(?)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

clf_SS = LogisticRegression()
clf_SS.fit(X_train, y_train)
y_pred = clf_SS.predict(X_test)
score_SS = clf_SS.score(X_test, y_test)
print(f"Model after normalization:\nAccuracy: {round(score_SS,2)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1: {round(f1_score(y_test, y_pred),2)}\n")
print(classification_report(y_test, clf_SS.predict(X_test)), "\n")
print(confusion_matrix(y_test, clf_SS.predict(X_test)), "\n")

# Build a model with the fit_intercept:
clf_fit = LogisticRegression(fit_intercept=True)
clf_fit.fit(X_train, y_train)
score_fit = clf_fit.score(X_test, y_test)
y_pred = clf_fit.predict(X_test)
print(f"Model with fit_intercept:\nAccuracy: {round(score_fit,2)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1: {round(f1_score(y_test, y_pred),2)}\n")
print(classification_report(y_test, clf_fit.predict(X_test)), "\n")
print(confusion_matrix(y_test, clf_fit.predict(X_test)), "\n")

# Build a model with a different solver:
clf_ll = LogisticRegression(solver="liblinear")
clf_ll.fit(X_train, y_train)
score_ll = clf_ll.score(X_test, y_test)
y_pred = clf_ll.predict(X_test)
print(f"Model with a different solver:\nAccuracy: {round(score_ll,2)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1: {round(f1_score(y_test, y_pred),2)}\n")
print(classification_report(y_test, clf_ll.predict(X_test)), "\n")
print(confusion_matrix(y_test, clf_ll.predict(X_test)), "\n")
