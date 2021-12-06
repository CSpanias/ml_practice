import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
from sklearn.metrics import recall_score, precision_recall_fscore_support, roc_curve


col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree',
    'age', 'label']

filepath = "C:/Users/10inm/Desktop/ml_practice/desicion_tree_datasets/diabetes.csv"
pima = pd.read_csv(filepath, skiprows = 1, header=None, names=col_names)
print(pima.head())
print(pima.shape)

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
clf = DecisionTreeClassifier(criterion="gini", max_depth=4, splitter="best")
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# using information gain
clf_ig = DecisionTreeClassifier(criterion="entropy", max_depth=3,
 splitter="best")
clf_ig = clf_ig.fit(X_train, y_train)
y_pred_ig = clf_ig.predict(X_test)

# using Random Forest
clf_rf = RandomForestClassifier(max_depth = 6)
clf_rf = clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

# using Logistic Regression
logreg = LogisticRegression(fit_intercept=True, solver='liblinear')
logreg = logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

print("Decision tree with gini index:\nAccuracy:", 
    round(accuracy_score(y_test, y_pred), 2))
print(f"Precision: {round(precision_score(y_test, y_pred), 2)}")
print(f"Recall: {round(recall_score(y_test, y_pred), 2)}")
print(f"F1: {round(f1_score(y_test, y_pred), 2)}\n")

print("Decision tree with information gain:\nAccuracy:", 
    round(accuracy_score(y_test, y_pred_ig), 2))
print(f"Precision: {round(precision_score(y_test, y_pred_ig), 2)}")
print(f"Recall: {round(recall_score(y_test, y_pred_ig), 2)}")
print(f"F1: {round(f1_score(y_test, y_pred_ig), 2)}\n")

print("Random Forest:\nAccuracy:", 
    round(accuracy_score(y_test, y_pred_rf), 2))
print(f"Precision: {round(precision_score(y_test, y_pred_rf), 2)}")
print(f"Recall: {round(recall_score(y_test, y_pred_rf), 2)}")
print(f"F1: {round(f1_score(y_test, y_pred_rf), 2)}\n")

print("Logistic Regression:\nAccuracy:", 
    round(accuracy_score(y_test, y_pred_logreg), 2))
print(f"Precision: {round(precision_score(y_test, y_pred_logreg), 2)}")
print(f"Recall: {round(recall_score(y_test, y_pred_logreg), 2)}")
print(f"F1: {round(f1_score(y_test, y_pred_logreg), 2)}\n")

"""
ROC CURVE
"""
print("ROC CURVE\n")
sensitivity_score = recall_score(y_test, y_pred_logreg)

print(precision_recall_fscore_support(y_test, y_pred_logreg),"\n")
"""
2nd array: RECALL

1st value: recall of negative class
2nd value: recall of positive class (=recall/sensitivity)
"""
def specificity_score(y_true, y_pred):
    """
    Calculating the specificity score, i.e., true negative rate,
    of a Logistic Regression model.
    """
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred_logreg)
    return r[0] # 2nd array, 1st value

print(f"Sensitivity: {sensitivity_score}")
print(f"Specificity: {specificity_score(y_test, y_pred_logreg)}\n")

# choose a different threshold besides 0.5
print(logreg.predict_proba(X_test)[:5],"\n")
"""
results in a 2D array with 2 values for each datapoint:
1st value: Proba of 0 class
2nd value: Proba of 1 class
"""
logreg.predict_proba(X_test)[:,1] # we only need the 2nd value

# compare the Proba values with another threshold (e.g. 0.75)
y_pred_logregA = logreg.predict_proba(X_test)[:,1] > 0.75

print(roc_curve(y_test, y_pred_logregA))
"""
returns an array of:
false positive rate (1-specificity, x-axis)
true positive rate (sensitivity, y-axis)
thresholds
"""
y_pred_proba = logreg.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1])

plt.plot(fpr,tpr)
plt.plot([0,1], [0,1], linestyle='--') # draw a line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title("ROC CURVE")
plt.xlabel("1-specificity")
plt.ylabel("sensitivity")
#plt.show()

print("\n'''\nAUC\n'''\n")

print(roc_auc_score(y_test, y_pred_proba[:,1]), "\n")

print("''''''''''''''''''''''''\nK-Fold Cross Validation\n''''''''''''''''''''''''\n")

kf = KFold(n_splits=5, shuffle=True)
# returns a generator, thus, we must convert it to a list with list()
splits = list(kf.split(X))
first_split = splits[0]
print(first_split)
# 1st array = train set indices, 2nd array test_set indices of the 1st split

"""
Use K-Fold to compare 3 different models:

1. A Logistic Regression with all features
2. A Logistic Regression with BMI and Age
3. A Logistic Regression with BMI, Age and BP.
"""
X1 = pd.DataFrame(pima[feature_cols])
kf = KFold(n_splits = 5, shuffle=True) # k, randomize order of data
X1 = X1.values
X2 = pd.DataFrame(pima)
X2 = X2[['bmi', 'age']].values
X3 = pd.DataFrame(pima)
X3 = X3[['bmi', 'age', 'bp']].values
y = pima['label']

def score_model(X, y, kf): 
    accuracy_scores = [] 
    precision_scores = [] 
    recall_scores = [] 
    f1_scores = [] 
    for train_index, test_index in kf.split(X): 
        X_train, X_test = X[train_index], X[test_index] 
        y_train, y_test = y[train_index], y[test_index] 
        model = LogisticRegression(solver='liblinear') 
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test) 
        accuracy_scores.append(accuracy_score(y_test, y_pred)) 
        precision_scores.append(precision_score(y_test, y_pred)) 
        recall_scores.append(recall_score(y_test, y_pred)) 
        f1_scores.append(f1_score(y_test, y_pred)) 
    print("accuracy:", np.mean(accuracy_scores)) 
    print("precision:", np.mean(precision_scores)) 
    print("recall:", np.mean(recall_scores)) 
    print("f1 score:", np.mean(f1_scores))


print("Logistic Regression with all features:")
print(score_model(X1, y, kf))
print("\nLogistic Regression with BMI and Age:")
print(score_model(X2, y, kf))
print("\nLogistic Regression with BMI, Age, and BP:")
print(score_model(X3, y, kf))
