import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()

print(iris.data.shape)
print(iris.target.shape)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, 
    test_size=0.25, random_state=0)

# 1st try: Default model
clf = LogisticRegression()
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(f"Model's default accuracy score: {round(score,2)}.")

# 2nd try: Model with Stardardization (everything in cm, thus, 
# won't change anything?)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

clf_SS = LogisticRegression()
clf_SS.fit(X_train, y_train)

score_SS = clf_SS.score(X_test, y_test)
print(f"Model with stardardization accuracy score: {round(score_SS,2)}.")

# 3rd try with fit_intercept:
clf_fit = LogisticRegression(fit_intercept=True)
clf_fit.fit(X_train, y_train)

score_fit = clf_fit.score(X_test, y_test)
print(f"Model with fit intercept accuracy score: {round(score_fit,2)}.")

# 4th try with different solver:
clf_ll = LogisticRegression(solver="liblinear", random_state=0)
clf_ll.fit(X_train, y_train)

score_ll = clf_ll.score(X_test, y_test)
print(f"Model with liblinear solver accuracy score: {round(score_ll,2)}.")

# 5th try with different solver (1):
clf_nc = LogisticRegression(solver="newton-cg", random_state=0)
clf_nc.fit(X_train, y_train)

score_nc = clf_nc.score(X_test, y_test)
print(f"Model with newton-cg solver accuracy score: {round(score_nc,2)}.")