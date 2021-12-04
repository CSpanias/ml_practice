import pandas as pd
from sklearn.tree import RandomForestClassifier
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

Documentation: 
scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
"""

# using Random Forest
clf_rf = RandomForestClassifier(n_estimators = 100)
clf_rf = clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

print("Accuracy (random forest): ", metrics.accuracy_score(y_test, y_pred_rf))