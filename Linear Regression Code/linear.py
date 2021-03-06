# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # comments in this guide will be added like this
import matplotlib.pyplot as plt 
import tensorflow.compat.v1 as tf
import pandas as pd
tf.disable_v2_behavior()


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

np.random.seed (101)
tf.set_random_seed (101) # sets the global random seed
# generate 100 random linear data points ranging from 0 to 25
x = np.linspace(0, 25, 100) 
y = np.linspace(0, 25, 100) 
# add noise to the random linear data 
x += np.random.uniform(-4, 4, 100) 
y += np.random.uniform(-4, 4, 100) 
n = len(x) # number of data points 

plt.scatter(x, y) # draws a scatter plot, plotting a single dot for each observation
plt.xlabel('x') # xlabel sets the label for the x-axis
plt.xlabel('y') # as this is lower than the label above, the horizontal axis only shows y
plt.title("Training Data") # sets the title for the plot
plt.show() # displays the figure

# in TensorFlow, variables are similar to standard coding variables that are initialized and can be modified later. Placeholders, on the other hand don’t require that initial value. It ‘reserves’ a block of memory for future use.

# define two placeholders
X = tf.placeholder(dtype=tf.float64) # placeholder X of type float
Y = tf.placeholder(dtype=tf.float64) # placeholder Y of type float

# declare two trainable TensorFlow variables and initialize them randomly using np.random.randn()
W = tf.Variable(np.random.randn(), name = "W", dtype=tf.float64)
b = tf.Variable(np.random.randn(), name = "b", dtype=tf.float64)

learning_rate = 0.01 # as seen earlier in the guide, learning rate is a hyperparameter controlling the scale on which the weights will be adjusted with respect to the loss
training_epochs = 10  # an epoch is the full iteration of the entire training data set

# building the hypothesis -> relationship between x and y
y_pred = tf.add(tf.multiply(X, W), b) # predicted y is the sum of (the product of X and W) and (b)

# Mean Squared Error Cost Function -> formula to determine value of the weight and bias from the given dataset
cost = tf.reduce_sum(tf.pow(y_pred-Y, 2)) / (2 * n)  # tf.pow works out the the power of one tensor with another similar to how in algebra xy would work -> then reduce_sum finds the sum of the elements across dimensions [1] A user on stack overflow has answered a question explaining how these dimensions work
  
# Gradient Descent Optimizer -> algorithm utilized to work out the optimized/ideal parameters
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 
  
# Global Variables Initializer 
init = tf.global_variables_initializer()

# Starting the Tensorflow Session 
with tf.Session() as sess: 
      
    # Initializing the Variables 
    sess.run(init) 
      
    # Iterating through all the epochs 
    for epoch in range(training_epochs): 
          
        # Feeding each data point into the optimizer using Feed Dictionary -> feed_dict feeds values into the placeholders
        for (_x, _y) in zip(x, y): # for…in zip merges the two lists here together   
           sess.run(optimizer, feed_dict = {X : _x, Y : _y}) 
          
        # Displaying the result after every 50 epochs 
        if (epoch + 1) % 50 == 0: 
            # Calculating the cost of every epoch 
            c = sess.run(cost, feed_dict = {X : x, Y : y}) 
            print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(W), "b =", sess.run(b)) 

        # Storing necessary values to be used outside the Session 
        training_cost = sess.run(cost, feed_dict ={X: x, Y: y}) 
        m = sess.run(W) # executing the graph with the value of W 
        c = sess.run(b) # executing the graph with the value of b

        predictions = m * x + c #y
print("Training cost =", training_cost, "M =", m, "C =", c, '\n')

# Plotting the Results 
plt.plot(x, y, 'ro', label ='Original data') # ro -> red circle with a label of Original data
plt.plot(x, predictions, label ='Fitted line') 
plt.title('Linear Regression Result') 
plt.legend() # displays small box containing description of the graph elements for example red dot denotes Original data
plt.show()

tf.sqrt(tf.reduce_mean(tf.squared_difference(y, y_pred)))

###################3

# y_true -> Y
# y_pred -> y_pred

TP = tf.count_nonzero(y_pred * Y)
TN = tf.count_nonzero((y_pred - 1) * (Y - 1))
FP = tf.count_nonzero(y_pred * (Y - 1))
FN = tf.count_nonzero((y_pred - 1) * Y)

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

TP = tf.get_static_value(TP)
TN = tf.get_static_value(TN)
FP = tf.get_static_value(FP)
FN = tf.get_static_value(FN)
precision = tf.get_static_value(precision)
recall = tf.get_static_value(recall)
f1 = tf.get_static_value(f1)

print (TP, TN, FP, FN, precision, recall, f1)

def perf_measure(Y, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if Y[i]==y_pred[i]==1:
           TP += 1
        if y_pred[i]==1 and Y[i]!=y_pred[i]:
           FP += 1
        if Y[i]==y_pred[i]==0:
           TN += 1
        if y_pred[i]==0 and Y[i]!=y_pred[i]:
           FN += 1

    return(TP, FP, TN, FN)
print ("TP: ", TP)

######
# changed float type to 64 because was getting dtype conflict and code is working now but im not
# getting the right answer