"""
This code is from a SVC tutorial from Prashant Banerjee on the Kaggle website.
https://www.kaggle.com/prashant111/svm-classifier-tutorial/notebook
"""
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

# set output to be automatically sized relative to the terminal size
pd.options.display.width = 0

# import dataset (the csv file comes with no headers)
# make a list with the required headers
# # IP = integrated profile, DM-SNR = delta-modulation and signal-to-noise ratio
header_list = ['IP Mean', 'IP Sd', 'IP Kurtosis', 'IP Skewness', 'DM-SNR Mean',
               'DM-SNR Sd', 'DM-SNR Kurtosis', 'DM-SNR Skewness', 'target_class']
# use the "names" attribute and assign the list of headers
df = pd.read_csv("C:\\Users\\10inm\\Desktop\\ml_practice\\svm_datasets\\HTRU_2.csv",
                 names=header_list)

# check column names, number of rows x cols, data types, missing values
print(df.info())

# check distribution of the target class
print(df['target_class'].value_counts())
# check the % distribution of the target class
print(df['target_class'].value_counts()/np.float(len(df)))

# check for outliers

# summary statistics
# compare 25%  & 75% to MIN & MAX = large difference indicates potential outliers
print(round(df.describe(),2))

# visual inspection
plt.figure(figsize=(12,10))

plt.subplot(4, 2, 1)
fig = df.boxplot(column='IP Mean')
fig.set_title('')
fig.set_ylabel('IP Mean')

plt.subplot(4, 2, 2)
fig = df.boxplot(column='IP Sd')
fig.set_title('')
fig.set_ylabel('IP Sd')

plt.subplot(4, 2, 3)
fig = df.boxplot(column='IP Kurtosis')
fig.set_title('')
fig.set_ylabel('IP Kurtosis')
plt.subplot(4, 2, 4)
fig = df.boxplot(column='IP Skewness')
fig.set_title('')
fig.set_ylabel('IP Skewness')

plt.subplot(4, 2, 5)
fig = df.boxplot(column='DM-SNR Mean')
fig.set_title('')
fig.set_ylabel('DM-SNR Mean')

plt.subplot(4, 2, 6)
fig = df.boxplot(column='DM-SNR Sd')
fig.set_title('')
fig.set_ylabel('DM-SNR Sd')

plt.subplot(4, 2, 7)
fig = df.boxplot(column='DM-SNR Kurtosis')
fig.set_title('')
fig.set_ylabel('DM-SNR Kurtosis')

plt.subplot(4, 2, 8)
fig = df.boxplot(column='DM-SNR Skewness')
fig.set_title('')
fig.set_ylabel('DM-SNR Skewness')
#plt.show()

"""
Two variants of SVMs:
1. Hard-margin --> does not tolerate outliers
2. Soft-margin --> allows incorrectly classified classes. We will have to 
adjust hyper-parameter C (= margin width) so it can ignore misclassified 
cases during training.
"""
# plot histogram to check distribution
plt.figure(figsize=(12,10))

plt.subplot(4, 2, 1)
fig = df['IP Mean'].hist(bins=20)
fig.set_xlabel('IP Mean')
fig.set_ylabel('Number of pulsar stars')

plt.subplot(4, 2, 2)
fig = df['IP Sd'].hist(bins=20)
fig.set_xlabel('IP Sd')
fig.set_ylabel('Number of pulsar stars')

plt.subplot(4, 2, 3)
fig = df['IP Kurtosis'].hist(bins=20)
fig.set_xlabel('IP Kurtosis')
fig.set_ylabel('Number of pulsar stars')

plt.subplot(4, 2, 4)
fig = df['IP Skewness'].hist(bins=20)
fig.set_xlabel('IP Skewness')
fig.set_ylabel('Number of pulsar stars')

plt.subplot(4, 2, 5)
fig = df['DM-SNR Mean'].hist(bins=20)
fig.set_xlabel('DM-SNR Mean')
fig.set_ylabel('Number of pulsar stars')

plt.subplot(4, 2, 6)
fig = df['DM-SNR Sd'].hist(bins=20)
fig.set_xlabel('DM-SNR Sd')
fig.set_ylabel('Number of pulsar stars')

plt.subplot(4, 2, 7)
fig = df['DM-SNR Kurtosis'].hist(bins=20)
fig.set_xlabel('DM-SNR Kurtosis')
fig.set_ylabel('Number of pulsar stars')

plt.subplot(4, 2, 8)
fig = df['DM-SNR Skewness'].hist(bins=20)
fig.set_xlabel('DM-SNR Skewness')
fig.set_ylabel('Number of pulsar stars')
#plt.show()

# assign X and y variables
X = df.drop(['target_class'], axis=1)
y = df['target_class']

# split train/test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0, shuffle=True)

# check the shape of X.train and X_test
print(X_train.shape, X_test.shape)

# Standardize our features
cols = X_train.columns
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
print(X_train.describe())

# assign the algorithm
# default parameters C=1.0, kernel=rbf, gamma=auto
svc = SVC()
svc.fit(X_train, y_train)

# make predictions on test set
y_pred = svc.predict(X_test)

# compute and print accuracy score
print("\nModel accuracy with rbf kernel and C=1.0 (default):",
      round(accuracy_score(y_test, y_pred),4))

# Increasing hyperparameter C --> fewer outliers
# instantiate classifier with rbf kernel and C=100
svc=SVC(C=100)
# assign X and y to classifier
svc.fit(X_train, y_train)
# predict on test set
y_pred = svc.predict(X_test)
# compute and print accuracy score
print("Model accuracy with rbf kernel and C=100:",
      round(accuracy_score(y_test, y_pred),4))

# instantiate classifier with rbf kernel and C=1000
svc=SVC(C=1000.0)
# fit classifier to training set
svc.fit(X_train,y_train)
# make predictions on test set
y_pred=svc.predict(X_test)
# compute and print accuracy score
print('Model accuracy score with rbf kernel and C=1000.0: {0:0.4f}'
      . format(accuracy_score(y_test, y_pred)))

# instantiate classifier with linear kernel and C=1.0
linear_svc=SVC(kernel='linear', C=1.0)
# fit classifier to training set
linear_svc.fit(X_train,y_train)
# make predictions on test set
y_pred_test=linear_svc.predict(X_test)
# compute and print accuracy score
print('Model accuracy score with linear kernel and C=1.0 : {0:0.4f}'
      . format(accuracy_score(y_test, y_pred_test)))

# instantiate classifier with linear kernel and C=100.0
linear_svc100=SVC(kernel='linear', C=100.0)
# fit classifier to training set
linear_svc100.fit(X_train, y_train)
# make predictions on test set
y_pred=linear_svc100.predict(X_test)
# compute and print accuracy score
print('Model accuracy score with linear kernel and C=100.0 : {0:0.4f}'
      . format(accuracy_score(y_test, y_pred)))

# instantiate classifier with linear kernel and C=1000.0
linear_svc1000=SVC(kernel='linear', C=1000.0)
# fit classifier to training set
linear_svc1000.fit(X_train, y_train)
# make predictions on test set
y_pred=linear_svc1000.predict(X_test)
# compute and print accuracy score
print('Model accuracy score with linear kernel and C=1000.0 : {0:0.4f}'
      . format(accuracy_score(y_test, y_pred)))

# check for overfitting, i.e. check train- and test-set accuracy
y_pred_train = linear_svc.predict(X_train)
print('Training-set accuracy score: {0:0.4f}'
      . format(accuracy_score(y_train, y_pred_train)))

# print the scores on training and test set
print('Training set score: {:.4f}'.format(linear_svc.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(linear_svc.score(X_test, y_test)))

"""
The model accuracy is 0.9832. But, we cannot say that our model is very 
good based on the above accuracy. We must compare it with the null accuracy. 

Null accuracy is the accuracy that could be achieved by always predicting the
most frequent class.
"""
# check class distribution in test set
print(y_test.value_counts())
"""
We can see that the occurences of most frequent class 0 is 3306.
"""
# check null accuracy score
null_accuracy = (3306/(3306+274))
print('Null accuracy score: {0:0.4f}'. format(null_accuracy))
"""
We can see that our model accuracy score is 0.9830 but null accuracy score 
is 0.9235. So, we can conclude that our SVM classifier is doing a very good 
job in predicting the class labels.
"""
# instantiate classifier with polynomial kernel and C=1.0
poly_svc=SVC(kernel='poly', C=1.0)
# fit classifier to training set
poly_svc.fit(X_train,y_train)
# make predictions on test set
y_pred=poly_svc.predict(X_test)
# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=1.0 : {0:0.4f}'
      . format(accuracy_score(y_test, y_pred)))

# instantiate classifier with polynomial kernel and C=100.0
poly_svc100=SVC(kernel='poly', C=100.0)
# fit classifier to training set
poly_svc100.fit(X_train, y_train)
# make predictions on test set
y_pred=poly_svc100.predict(X_test)
# compute and print accuracy score
print('Model accuracy score with polynomial kernel and C=100 : {0:0.4f}'
      . format(accuracy_score(y_test, y_pred)))

# instantiate classifier with sigmoid kernel and C=1.0
sigmoid_svc=SVC(kernel='sigmoid', C=1.0)
# fit classifier to training set
sigmoid_svc.fit(X_train,y_train)
# make predictions on test set
y_pred=sigmoid_svc.predict(X_test)
# compute and print accuracy score
print('Model accuracy score with sigmoid kernel and C=1.0 : {0:0.4f}'
      . format(accuracy_score(y_test, y_pred)))

# instantiate classifier with sigmoid kernel and C=100.0
sigmoid_svc100=SVC(kernel='sigmoid', C=100.0)
# fit classifier to training set
sigmoid_svc100.fit(X_train,y_train)
# make predictions on test set
y_pred=sigmoid_svc100.predict(X_test)
# compute and print accuracy score
print('Model accuracy score with sigmoid kernel and C=100.0 : {0:0.4f}'
      . format(accuracy_score(y_test, y_pred)))

"""
We get maximum accuracy with rbf and linear kernel with C=100.0. and the 
accuracy is 0.9832. Based on the above analysis we can conclude that our 
classification model accuracy is very good. Our model is doing a very good 
job in terms of predicting the class labels.

But, this is not true. Here, we have an imbalanced dataset. The problem is 
that accuracy is an inadequate measure for quantifying predictive performance 
in the imbalanced dataset problem.

So, we must explore alternative metrics that provide better guidance in 
selecting models. In particular, we would like to know the underlying 
distribution of values and the type of errors our classifier is making.

One such metric to analyze the model performance in imbalanced classes 
problem is Confusion matrix.

*
FP = Type 1 Error
FN = Type 2 Error
"""
# Print the Confusion Matrix and slice it into four pieces
cm = confusion_matrix(y_test, y_pred_test)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])

# visualize confusion matrix with seaborn heatmap
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1',
                                           'Actual Negative:0'],
                                 index=['Predict Positive:1',
                                        'Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
#plt.show()

print(classification_report(y_test, y_pred_test))

# plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Predicting a Pulsar Star classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()

# compute ROC AUC
ROC_AUC = roc_auc_score(y_test, y_pred_test)
print('ROC AUC : {:.4f}'.format(ROC_AUC))

# calculate cross-validated ROC AUC
Cross_validated_ROC_AUC = cross_val_score(linear_svc, X_train, y_train, cv=10,
                                          scoring='roc_auc').mean()
print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))

"""
k-fold cross-validation is a very useful technique to evaluate model 
performance. But, it fails here because we have a imbalanced dataset. 
So, in the case of imbalanced dataset, I will use another technique to 
evaluate model performance. It is called stratified k-fold cross-validation.

In stratified k-fold cross-validation, we split the data such that the
proportions between classes are the same in each fold as they are in the 
whole dataset.

Moreover, I will shuffle the data before splitting because shuffling yields 
much better result.
"""
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
linear_svc = SVC(kernel='linear')
linear_scores = cross_val_score(linear_svc, X, y, cv=kfold)
# print cross-validation scores with linear kernel
print('Stratified cross-validation scores with linear kernel:',
      round(linear_scores, 4))
# print average cross-validation score with linear kernel
print('\nAverage stratified cross-validation score with linear kernel:',
      round(linear_scores.mean(), 2))

rbf_svc=SVC(kernel='rbf')
rbf_scores = cross_val_score(rbf_svc, X, y, cv=kfold)
# print cross-validation scores with rbf kernel
print('\nStratified Cross-validation scores with rbf kernel:',
      round(rbf_scores, 4))

# print average cross-validation score with rbf kernel
print(f'Average stratified cross-validation score with rbf kernel:',
      round(rbf_scores.mean(), 4))

"""
I obtain higher average stratified k-fold cross-validation score of 0.9789 with 
linear kernel but the model accuracy is 0.9832. So, stratified cross-validation
technique does not help to improve the model performance.
"""
# Hyperparameter Optimization using GridSearch CV
# instantiate classifier with default hyperparameters with kernel=rbf,
# C=1.0 and gamma=auto
svc = SVC()
# declare parameters for hyperparameter tuning
parameters = [ {'C':[1, 10, 100, 1000], 'kernel':['linear']},
               {'C':[1, 10, 100, 1000], 'kernel':['rbf'],
                'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
               {'C':[1, 10, 100, 1000], 'kernel':['poly'], 'degree': [2,3,4] ,
                'gamma':[0.01,0.02,0.03,0.04,0.05]}
              ]
grid_search = GridSearchCV(estimator = svc,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)
grid_search.fit(X_train, y_train)

"""
examine the best model
"""
# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n'.format(grid_search.best_score_))
# print parameters that give the best results
print('Parameters that give the best results :','\n',
      (grid_search.best_params_))
# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n',
      (grid_search.best_estimator_))

# calculate GridSearch CV score on test set
print('GridSearch CV score on test set: {0:0.4f}'
      .format(grid_search.score(X_test, y_test)))

"""
Our original model test accuracy is 0.9832 while GridSearch CV score on 
test-set is 0.9835.

So, GridSearch CV helps to identify the parameters that will improve the 
performance for this particular model.

Here, we should not confuse best_score_ attribute of grid_search with the 
score method on the test-set.

The score method on the test-set gives the generalization performance 
of the model. Using the score method, we employ a model trained on the 
whole training set.
The best_score_ attribute gives the mean cross-validation accuracy, 
with cross-validation performed on the training set.
"""

"""
RESULTS AND CONCLUSIONS

There are outliers in our dataset. So, as I increase the value of C to limit 
fewer outliers, the accuracy increased. This is true with different kinds of 
kernels.

We get maximum accuracy with rbf and linear kernel with C=100.0 and the 
accuracy is 0.9832. So, we can conclude that our model is doing a very good 
job in terms of predicting the class labels. But, this is not true. Here, 
we have an imbalanced dataset. Accuracy is an inadequate measure for 
quantifying predictive performance in the imbalanced dataset problem. 
So, we must explore confusion matrix that provide better guidance in 
selecting models.

ROC AUC of our model is very close to 1. So, we can conclude that our classifier does a good job in classifying the 
pulsar star.

I obtain higher average stratified k-fold cross-validation score of 0.9789 
with linear kernel but the model accuracy is 0.9832. 
So, stratified cross-validation technique does not help to improve 
the model performance.

Our original model test accuracy is 0.9832 while GridSearch CV score on
test-set is 0.9835. So, GridSearch CV helps to identify the parameters 
that will improve the performance for this particular model.
"""
