"""
Support-vector Machine using sklearn.

Provided as an example from the course.

Intented to be used on the Kaggle website.
"""
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn import svm

x = np.array([1,5,1.5,8,1,9,7,8.7,2.3,5.5,7.7,6.1])
y = np.array([2,8,1.8,8,0.6,11,10,9.4,4,3,8.8,7.5])

plt.scatter(x, y)
plt.show()

training_x = np.vstack((x, y)).T
training_y = [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1]

clf = svm.SVC(kernel='linear', C=1.0)

clf.fit(training_x, training_y)

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(0, 13)
yy = a * xx - clf.intercept_[0] / w[1]
plt.plot(xx, yy, 'k-')
plt.scatter(training_x[:, 0], training_x[:, 1 ], c=training_y)
plt.legend()
plt.show()