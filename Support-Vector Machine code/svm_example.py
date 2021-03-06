"""
Support-vector Machine using sklearn.

Provided as an example from the course.

Intented to be used on the Kaggle website.
"""
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from matplotlib import pyplot as plt
import numpy as np
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


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session