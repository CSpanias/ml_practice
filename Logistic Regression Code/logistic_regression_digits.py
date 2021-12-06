"""
Logistic regression using sklearn.

Built five different models.

Compared the accuracy among them.
"""
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings('ignore')

# load digits dataset
digits = load_digits()

# check the dataset
print(digits.data.shape)
print(digits.target.shape)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, 
    test_size=0.25, random_state=0)

# Build a model with default settings (*increased iterations)
clf = LogisticRegression(max_iter=2500)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(f"Model's default accuracy: {round(score,2)}.")

# Build a model with normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

clf_SS = LogisticRegression()
clf_SS.fit(X_train, y_train)
score_SS = clf_SS.score(X_test, y_test)
print(f"Model with stardardization accuracy score: {round(score_SS,2)}.")

# Build a model with fit_intercept:
clf_fit = LogisticRegression(fit_intercept=True)
clf_fit.fit(X_train, y_train)
score_fit = clf_fit.score(X_test, y_test)
print(f"Model with fit intercept accuracy score: {round(score_fit,2)}.")

# Build a model with a different solver:
clf_ll = LogisticRegression(solver="liblinear", random_state=0)
clf_ll.fit(X_train, y_train)
score_ll = clf_ll.score(X_test, y_test)
print(f"Model with liblinear solver accuracy score: {round(score_ll,2)}.")

# Build a model with a different solver (1):
clf_nc = LogisticRegression(solver="newton-cg", random_state=0)
clf_nc.fit(X_train, y_train)
score_nc = clf_nc.score(X_test, y_test)
print(f"Model with newton-cg solver accuracy score: {round(score_nc,2)}.")
