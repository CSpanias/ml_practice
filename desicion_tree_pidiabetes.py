import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
from sklearn.ensemble import RandomForestClassifier
# scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.model_selection import train_test_split
from sklearn import metrics

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree',
    'age', 'label']

filepath = "C:/Users/10inm/Desktop/ML practice/desicion_tree_datasets/diabetes.csv"
pima = pd.read_csv(filepath, skiprows = 1, header=None, names=col_names)
print(pima.head())

feature_cols = ['pregnant', 'glucose', 'bp', 'insulin', 'bmi', 'pedigree', 'age']

X = pima[feature_cols]
y = pima['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
    random_state = 1)

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
clf_ig = DecisionTreeClassifier(criterion="entropy", max_depth=3, splitter="best", random_state=40)
clf_ig = clf_ig.fit(X_train, y_train)
y_pred_ig = clf_ig.predict(X_test)

# using Random Forest
clf_rf = RandomForestClassifier(max_depth = 6, random_state = 40)
clf_rf = clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

print("Accuracy score (decision tree + gini index):\n", metrics.accuracy_score(y_test, y_pred))
print("Accuracy score (decision tree + information gain):\n", metrics.accuracy_score(y_test, y_pred_ig))
print("Accuracy score (random forest):\n", metrics.accuracy_score(y_test, y_pred_rf))