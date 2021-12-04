# import required modules
import pandas as pd
import numpy as np
import category_encoders as ce
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree

# import dataset (notice the use of "/" instead of "\" !!!)
filepath = "C:/Users/10inm/Desktop/ML practice/desicion_tree_datasets/mushrooms.csv"
mushrooms = pd.read_csv(filepath)
# check the first 5 rows of the dataset
print(mushrooms.head())
# check the data types of columns
# object = strings/characters, we will need to convert them in numbers
print(mushrooms.dtypes)
# check how many different classes we have
print(mushrooms['class'].unique())

# select which features we want to use (23 in total)
feature_cols = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring', 
    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
    'ring-type', 'spore-print-color', 'population', 'habitat']

X = mushrooms[feature_cols].values
y = mushrooms['class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
    random_state = 1)

encoder = ce.OrdinalEncoder(X)
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

"""
Choose the attribute selection measure:
"gini" (default) = gini index 
"entropy" = information gain

Choose the split strategy: 
"best" = best split (default)
"random" = best random split

Choose maximum depth:
integer = higher value of max depth cause overfitting and lower value
causes underfitting
"None" = nodes are expanded until all the leaves contain less than
 min_samples_split samples (default)
"""

# using gini index
clf = DecisionTreeClassifier(criterion="gini", max_depth=4, splitter="best", random_state=40)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# using information gain
clf_ig = DecisionTreeClassifier(criterion="entropy", splitter="best", random_state=40)
clf_ig = clf_ig.fit(X_train, y_train)
y_pred_ig = clf_ig.predict(X_test)

# using Random Forest
clf_rf = RandomForestClassifier(max_depth = 6, random_state = 40)
clf_rf = clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

print("Accuracy score (decision tree + gini index):\n", metrics.accuracy_score(y_test, y_pred))
print("Accuracy score (decision tree + information gain):\n", metrics.accuracy_score(y_test, y_pred_ig))
print("Accuracy score (random forest):\n", metrics.accuracy_score(y_test, y_pred_rf))


tree.plot_tree(clf,
              feature_names = feature_cols,
              class_names = ['p', 'e'],
              filled = True)

tree.export_graphviz(clf,
                     out_file="tree.dot",
                     feature_names = feature_cols, 
                     class_names= ['p', 'e'],
                     filled = True)

tree.plot_tree(clf_ig,
              feature_names = feature_cols,
              class_names = ['p', 'e'],
              filled = True)

tree.export_graphviz(clf_ig,
                     out_file="tree_ig.dot",
                     feature_names = feature_cols, 
                     class_names= ['p', 'e'],
                     filled = True)

