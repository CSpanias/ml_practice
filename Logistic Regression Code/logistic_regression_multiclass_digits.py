"""
Logistic Regression using sklearn.

Built four different models.

Compared the accuracy score among them.
"""
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')


filepath = "C:\\Users\\10inm\\Desktop\\ml_practice\\logistic_regression_datasets\\modifiedDigits4Classes.csv"
df = pd.read_csv(filepath)
print(df.head())
print(df.shape, "\n")

pixel_colnames = df.columns[:-1]

X_train, X_test, y_train, y_test = train_test_split(df[pixel_colnames],
  df['label'], random_state=0)

# Build a model with default settings (*increased iterations)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(f"Model's default accuracy score: {score}.")

# Build a model with normalization
# everything in cm, thus, won't change much(?)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

clf_SS = LogisticRegression()
clf_SS.fit(X_train, y_train)
score_SS = clf_SS.score(X_test, y_test)
print(f"Model with stardardization accuracy score: {score_SS}.")

# Build a model with fit_intercept:
clf_fit = LogisticRegression(fit_intercept=True)
clf_fit.fit(X_train, y_train)
score_fit = clf_fit.score(X_test, y_test)
print(f"Model with fit intercept accuracy score: {score_fit}.")

# Build a model with a different solver:
clf_ll = LogisticRegression(solver="liblinear",multi_class='ovr', 
    random_state=0)
clf_ll.fit(X_train, y_train)
score_ll = clf_ll.score(X_test, y_test)
print(f"Model with liblinear solver accuracy score: {score_ll}.")