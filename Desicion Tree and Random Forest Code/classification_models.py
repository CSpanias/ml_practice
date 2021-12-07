# import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import preprocessing

import warnings
warnings.filterwarnings('ignore')
pd.options.display.width = 0
# import dataset
df = pd.read_csv("C:/Users/10inm/Desktop/ml_practice/desicion_tree_datasets"
                 "/diabetes.csv")

# check shape, datatype, NaNs
print(df.info())

# check dataset
print(df.head())

# exploratory data analysis (EDA): correlation and heatmap
df_corr = df.corr()
#print(df_corr)
sns.heatmap(df_corr, annot=True,cmap='coolwarm')
#plt.show()

# EDA: pairplot
sns.pairplot(df)
#plt.show()

# assign X and y variables
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# normalize features
X = preprocessing.normalize(X)
print("Normalized Data = ", X)

# split training/testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                shuffle=True, random_state=10)
# assign model
logreg = LogisticRegression()
dectree = DecisionTreeClassifier()
rdfor = RandomForestClassifier()
svm = svm.SVC()
nbg = GaussianNB()

# train model
logreg.fit(X_train, y_train)
dectree.fit(X_train, y_train)
rdfor.fit(X_train, y_train)
svm.fit(X_train, y_train)
nbg.fit(X_train, y_train)

# predict test data
y_predlog = logreg.predict(X_test)
y_preddec = dectree.predict(X_test)
y_predrdf = rdfor.predict(X_test)
y_predsvm = svm.predict(X_test)
y_prednbg = nbg.predict(X_test)

# evalute
print("\nLogistic Regression:",
      "\nAccuracy:", round(metrics.accuracy_score(y_test, y_predlog),2),
      "\nPrecision:", round(metrics.precision_score(y_test, y_predlog),2),
      "\nRecall:", round(metrics.recall_score(y_test, y_predlog),2),
      "\nF1:", round(metrics.f1_score(y_test, y_predlog),2))
print("\nDecision Tree:",
      "\nAccuracy:", round(metrics.accuracy_score(y_test, y_preddec),2),
      "\nPrecision:", round(metrics.precision_score(y_test, y_preddec),2),
      "\nRecall:", round(metrics.recall_score(y_test, y_preddec),2),
      "\nF1:", round(metrics.f1_score(y_test, y_preddec),2))
print("\nRandom Forest:",
      "\nAccuracy:", round(metrics.accuracy_score(y_test, y_predrdf),2),
      "\nPrecision:", round(metrics.precision_score(y_test, y_predrdf),2),
      "\nRecall:", round(metrics.recall_score(y_test, y_predrdf),2),
      "\nF1:", round(metrics.f1_score(y_test, y_predrdf),2))
print("\nSupport Vector Machine:",
      "\nAccuracy:", round(metrics.accuracy_score(y_test, y_predsvm),2),
      "\nPrecision:", round(metrics.precision_score(y_test, y_predsvm),2),
      "\nRecall:", round(metrics.recall_score(y_test, y_predsvm),2),
      "\nF1:", round(metrics.f1_score(y_test, y_predsvm),2))
print("\nGaussian Naive-Bayes:"
      "\nAccuracy:", round(metrics.accuracy_score(y_test, y_prednbg), 2),
      "\nPrecision:", round(metrics.precision_score(y_test, y_prednbg), 2),
      "\nRecall:", round(metrics.recall_score(y_test, y_prednbg), 2),
      "\nF1:", round(metrics.f1_score(y_test, y_prednbg), 2))

