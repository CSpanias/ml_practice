import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

file_path = "C:\\Users\\10inm\\Desktop\\ML practice\\desicion_tree_datasets\\drug200.csv"
df = pd.read_csv(file_path)

print(df.head())
print(df.shape)
print(df['Drug'].unique())

feature_cols = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = df['Drug'].values

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
    test_size = 0.20, random_state = 0)

"""
decision trees like most machine learning algorithms cannot process 
labeled data values hence the data is encoded as numerical data is 
easily handled by this algorithm. A stackoverflow question expands 
on this topic.
"""

import category_encoders as ce
encoder = ce.OrdinalEncoder(X)
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

print(X_train)

clf_gini = DecisionTreeClassifier(criterion = 'gini', 
    max_depth = 10, random_state = 0)
clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)

clf_info_gain = DecisionTreeClassifier(criterion = 'entropy', 
    max_depth = 10, random_state = 0)
clf_info_gain.fit(X_train, y_train)
y_pred_info_gain = clf_info_gain.predict(X_test)

print('Model accuracy score with criterion gini index: ' 
    f'{accuracy_score(y_test, y_pred_gini)}')
print('Model accuracy score with criterion information gain: ' 
    f'{accuracy_score(y_test, y_pred_info_gain)}')
