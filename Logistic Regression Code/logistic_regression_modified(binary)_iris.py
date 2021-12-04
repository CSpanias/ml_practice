import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

filepath = "C:\\Users\\10inm\\Desktop\\ml_practice\\logistic_regression_datasets\\modifiedIris2Classes.csv"
df = pd.read_csv(filepath)
print(df.shape)
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df[['petal length (cm)']], 
df['target'], test_size=0.25, random_state=0)

# 1st try: Default model
clf = LogisticRegression()
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(f"Model's default accuracy score: {score}.")

# 2nd try: Model with Stardardization (everything in cm, thus, 
# won't change anything?)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

clf_SS = LogisticRegression()
clf_SS.fit(X_train, y_train)

score_SS = clf_SS.score(X_test, y_test)
print(f"Model with stardardization accuracy score: {score_SS}.")

# 3rd try with fit_intercept:
clf_fit = LogisticRegression(fit_intercept=True)
clf_fit.fit(X_train, y_train)

score_fit = clf_fit.score(X_test, y_test)
print(f"Model with fit intercept accuracy score: {score_fit}.")

# 4th try with different solver:
clf_ll = LogisticRegression(solver="liblinear")
clf_ll.fit(X_train, y_train)

score_ll = clf_ll.score(X_test, y_test)
print(f"Model with liblinear solver accuracy score: {score_ll}.")
"""
classification_report(y_test, clf.predict(X_test))

# confusion matrix
cm = confusion_matrix(y_test, clf.predict(X_test))

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True,
    fmt=".0f",
    linewidths=.5,
    square=True,
    cmap = "Blues");
plt.ylabel("Actual label", fontsize = 17);
plt.xlabel("Predicted label", fontsize = 17);
plt.title('Accuracy Score: {}'.format(score), size = 17);
plt.tick_params(labelsize=15)
#plt.show()


# decision boundary
example_df = pd.DataFrame()
example_df.loc[:, 'petal length (cm)'] = X_test.reshape(-1)
example_df.loc[:, 'target'] = y_test.values
example_df['logistic_preds'] = pd.DataFrame(clf.predict_proba(X_test))[1]


fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,7));


virginicaFilter = example_df['target'] == 1
versicolorFilter = example_df['target'] == 0

ax.scatter(example_df.loc[virginicaFilter, 'petal length (cm)'].values,
            example_df.loc[virginicaFilter, 'logistic_preds'].values,
           color = 'g',
           s = 60,
           label = 'virginica')


ax.scatter(example_df.loc[versicolorFilter, 'petal length (cm)'].values,
            example_df.loc[versicolorFilter, 'logistic_preds'].values,
           color = 'b',
           s = 60,
           label = 'versicolor')

ax.axhline(y = .5, c = 'y')

ax.axhspan(.5, 1, alpha=0.05, color='green')
ax.axhspan(0, .4999, alpha=0.05, color='blue')
ax.text(0.5, .6, 'Classified as viginica', fontsize = 16)
ax.text(0.5, .4, 'Classified as versicolor', fontsize = 16)

ax.set_ylim(0,1)
ax.legend(loc = 'lower right', markerscale = 1.0, fontsize = 12)
ax.tick_params(labelsize = 18)
ax.set_xlabel('petal length (cm)', fontsize = 24)
ax.set_ylabel('probability of virginica', fontsize = 24)
ax.set_title('Logistic Regression Predictions', fontsize = 24)
fig.tight_layout()
#plt.show()
"""