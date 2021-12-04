import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

# import dataset
filepath = "C:\\Users\\10inm\\Desktop\\ML practice\\linear_regression_datasets\\life_expectancy.csv"
df = pd.read_csv(filepath, skipinitialspace=True)
df = df.rename(columns=lambda x: x.strip())

# check dataset
print(df.info())
print(df.head())
print(df.isnull().sum())
df = df.dropna()
print(df.isnull().sum())


X = df[['Status', 'Adult Mortality', 'infant deaths', 'Alcohol', 'HIV/AIDS',
    'GDP', 'Schooling', 'under-five deaths', 'thinness  1-19 years',
     'thinness 5-9 years', 'Population', 'Diphtheria', 'Total expenditure',
     'percentage expenditure', 'Hepatitis B', 'Measles', 'Polio',
     'Income composition of resources']]
y = df['Life expectancy'].values

X = pd.get_dummies(data=X, drop_first=True)
X = X.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
                                                    random_state = 40)

# First try with default settings

model = LinearRegression(fit_intercept=False)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

# Second try with changed parameters

model_fit = LinearRegression(fit_intercept=True)
model_fit.fit(X_train, y_train)
score_fit = model_fit.score(X_test, y_test)

# Third try with changed parameters 2

model_fit_norm = LinearRegression(fit_intercept=True, normalize=True)
model_fit_norm.fit(X_train, y_train)
score_fit_norm = model_fit_norm.score(X_test, y_test)

# Fourth try with StandardScaler

from sklearn.preprocessing import StandardScaler
"""
Variables that are measured at different scales do not contribute
equally to the  model fitting & model learned function and might end up 
creating a bias.

StandardScaler() will normalize the features i.e. each column of X,
INDIVIDUALLY so that each column/feature/variable will have μ = 0 and σ = 1.

Standardization is only applicable on the data values that follows
Normal Distribution!
"""
scaler = StandardScaler()
# Fit on training set only.
print(f"Before Standardization:\n{X}")
X = scaler.fit_transform(X)
print(f"After Standardization:\n{X}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
                                                    random_state = 40)
# Apply transform to both the training and test set.
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

model_SS = LinearRegression()
model_SS.fit(X_train, y_train)
score_SS = model_SS.score(X_test, y_test)

print(f"The default model's accuracy is: {round(score,2)}.")
print(f"The model's accuracy with fit_intercept is: {round(score_fit,2)}.")
print("The model's accuracy with fit_intercept plus normalization "
f"is: {round(score_fit_norm,2)}.")
print(f"The model's accuracy with StandardScaler is: {round(score_SS,2)}.")

prediction = model_fit.predict(X_test)
sns.regplot(y_test, prediction)
#plt.text(0.35, 0.9, f"intecept = {intercept}\ncoef = {coef}", fontsize=7)
plt.show()