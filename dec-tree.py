# comments in this guide will be added like this

# The import keyword lets you ‘bind’ the library/package to a variable. This is to avoid naming collisions and shorten long names/titles into shorter ones that’d be easier to use. Numpy is imported for ease in creating arrays, TensorFlow will be utilized for the regression itself, Pandas is to interact with the DataFrame [1]  and, lastly, MatplotLib supports data plotting

import numpy as np # to utilize its linear algebra
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # statistical visualization of data
import pandas as pd

from sklearn.model_selection import train_test_split # Import train_test_split function [2]
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.metrics import accuracy_score #Import scikit-learn metrics module to calculate accuracy (of the Decision Tree model built in this guide)

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/kaggle/input/car-evaluation-data-set/car_evaluation.csv', header=None) # read_csv is a pandas function utilized to read csv files and do operations on it 

# renaming column names
col_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"] 
df.columns = col_names # assigning the above column names to the dataframe’s current columns

df.shape # viewing dataset’s dimension
df.head(5) # previewing the first 5 rows of the dataset

# splitting the dataset into features and target variable
feature_cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
X = df.drop(['class'], axis=1) # feature matrix [1]
y = df['class'] # target variable 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) # 67% training and 33% test // found out by 1 - test_size = 1 - 0.33 = 0.67 -> 67%
X_train.shape, X_test.shape

# decision trees like most machine learning algorithms cannot process labeled data values hence the data is encoded as numerical data is easily handled by this algorithm. A stackoverflow question expands on this topic.

import category_encoders as ce

encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
X_train.head()

X_test.head()

# Create Decision Tree classifier object ; clf refers to classifier
clf_gini = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, random_state = 0) 
clf_gini.fit(X_train, y_train) 
y_pred_gini = clf_gini.predict(X_test) 

# determining model accuracy i.e. how often is the classifier correct?
print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))

plt.figure(figsize=(12,8))
from sklearn import tree
tree.plot_tree(clf_gini.fit(X_train, y_train))

import graphviz # graphviz takes description of graphs and data and constructs diagrams based off of that information
dot_data = tree.export_graphviz(clf_gini, out_file=None, 
                              feature_names=X_train.columns,  
                              class_names=y_train,  
                              filled=True, rounded=True,  
                              special_characters=True)
graph = graphviz.Source(dot_data) 
graph 
