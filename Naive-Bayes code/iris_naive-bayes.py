"""
Naive-Bayes using sklearn.

Used all but one* NB variants.

*out-of-core NB
"""
import numpy as np 
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB, ComplementNB, MultinomialNB, BernoulliNB

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
 random_state=0)

# Build a GaussianNB model
model = GaussianNB()
y_pred = model.fit(X_train, y_train).predict(X_test)
print("GaussianNB\nNumber of mislabeled points out of a total %d points : %d\n"
      % (X_test.shape[0], (y_test != y_pred).sum()))

# Build a CategoricalNB model (used mainly for categorical distributed data)
model = CategoricalNB()
y_pred = model.fit(X_train, y_train).predict(X_test)
print("CategoricalNB\nNumber of mislabeled points out of a total %d points : %d\n"
      % (X_test.shape[0], (y_test != y_pred).sum()))

# Build a ComplementNB model (used mainly for text classification)
model = ComplementNB()
y_pred = model.fit(X_train, y_train).predict(X_test)
print("ComplementNB\nNumber of mislabeled points out of a total %d points : %d\n"
      % (X_test.shape[0], (y_test != y_pred).sum()))

# Build a MultinomialNB model (used mainly for text classification)
model = MultinomialNB()
y_pred = model.fit(X_train, y_train).predict(X_test)
print("MultinomialNB\nNumber of mislabeled points out of a total %d points : %d\n"
      % (X_test.shape[0], (y_test != y_pred).sum()))

"""
Build a BernoulliNB model (used for data that is distributed according to 
multivariate Bernoulli distributions; i.e., there may be multiple features 
but each one is assumed to be a binary-valued (Bernoulli, boolean) variable.
"""
model = BernoulliNB()
y_pred = model.fit(X_train, y_train).predict(X_test)
print("BernoulliNB\nNumber of mislabeled points out of a total %d points : %d\n"
      % (X_test.shape[0], (y_test != y_pred).sum()))