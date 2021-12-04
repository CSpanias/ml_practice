import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


filepath = "C:\\Users\\10inm\\Desktop\\ML practice\\logistic_regression_datasets\\modifiedDigits4Classes.csv"
df = pd.read_csv(filepath)
print(df.head())
print(df.shape)

pixel_colnames = df.columns[:-1]

X_train, X_test, y_train, y_test = train_test_split(df[pixel_colnames],
  df['label'], random_state=0)

# 1st try: Default model
clf = LogisticRegression(max_iter=1000)
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
clf_ll = LogisticRegression(solver="liblinear",multi_class='ovr', 
    random_state=0)
clf_ll.fit(X_train, y_train)

score_ll = clf_ll.score(X_test, y_test)
print(f"Model with liblinear solver accuracy score: {score_ll}.")

"""
image_values = df.loc[0, pixel_colnames].values

plt.figure(figsize=(10,2))
for index in range(0, 4):

    plt.subplot(1, 5, 1 + index )
    image_values = df.loc[index, pixel_colnames].values
    image_label = df.loc[index, 'label']
    plt.imshow(image_values.reshape(8,8), cmap ='gray')
    plt.title('Label: ' + str(image_label))
plt.show()

print('Training accuracy:', clf.score(X_train, y_train))
print('Test accuracy:', clf.score(X_test, y_test))

print(clf.intercept_)
print(clf.coef_.shape)

print(clf.predict_proba(X_test[0:1]))
print(clf.predict(X_test[0:1]))
"""