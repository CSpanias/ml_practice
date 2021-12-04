"""
 The import keyword lets you bind the library/package to a
variable. This is to avoid naming collisions and shorten long
names/titles into shorter ones that would be easier to use. Numpy
is imported for ease in creating arrays , TensorFlow will be
utilized for the regression itself and , lastly , MatplotLib
supports data plotting
"""

import numpy as np 
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior ()

"""
in order to make random numbers predictable , seeds are set to be
reproducible by setting them at a value of 101 so the generator
creating a random value for the seed value (101) will always be
the same random value
"""
np.random.seed (101)
# sets the global random seed
tf.set_random_seed (101)
# generate 100 random linear data points ranging from 0 to 25
x = np.linspace(0, 25, 100)
y = np.linspace(0, 25, 100)
# add noise to the random linear data
x += np.random.uniform(-4, 4, 100)
y += np.random.uniform(-4, 4, 100)
n = len(x) # number of data points

# draws a scatter plot , plotting a single dot for each observation
plt.scatter(x, y)
# xlabel sets the label for the x-axis
plt.xlabel("x")
# as this is lower than the label above , the horizontal axis only shows y
plt.xlabel("y")
# sets the title for the plot
plt.title("Training Data")
# displays the figure
plt.show()

"""
in TensorFlow , variables are similar to standard coding variables
that are initialized and can be modified later. Placeholders , on
the other hand d o n t require that initial value. It
reserves a block of memory for future use
"""
# define two placeholders
X = tf.placeholder(dtype=tf.float64) # placeholder X of type float
Y = tf.placeholder(dtype=tf.float64) # placeholder Y of type float

""" declare two trainable TensorFlow variables and initialize them 
randomly using np.random.randn ()"""
# the variable W denotes weight
W = tf.Variable(np.random.randn (), name = "W", dtype=tf.float64)
# the variable b denotes -> bias
b = tf.Variable(np.random.randn (), name = "b", dtype=tf.float64)

"""
as seen earlier in the guide , learning rate is a hyperparameter 
controlling the scale on which the weights will be adjusted with respect 
to the loss 
"""
learning_rate = 0.01 
# an epoch is the full iteration of the entire training data set
training_epochs = 1000

# building the hypothesis -> relationship between x and y
# predicted y is the sum of (the product of X and W) and (b)
y_pred = tf.add(tf.multiply(X, W), b)
"""
Mean Squared Error Cost Function -> formula to determine value of
the weight and bias from the given dataset """
"""
tf.pow works out the the power of one tensor with another similar to how in
algebra xy would work -> then reduce_sum finds the sum of the elements across 
dimensions.
"""
cost = tf.reduce_sum(tf.pow(y_pred -Y, 2)) / (2 * n) 
# Gradient Descent Optimizer -> algorithm utilized to work out the
# optimized/ideal parameters
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Global Variables Initializer
init = tf.global_variables_initializer()

# Starting the Tensorflow Session
with tf.Session () as sess:
    # Initializing the Variables
    sess.run(init)

# Iterating through all the epochs
for epoch in range(training_epochs):

    # Feeding each data point into the optimizer using Feed Dictionary -> 
    # feed_dict feeds values into the placeholders

    # for in zip merges the two lists here together
    for (_x , _y) in zip(x, y):
        sess.run(optimizer , feed_dict = {X : _x , Y : _y})

    # Displaying the result after every 50 epochs
    if (epoch + 1) % 50 == 0:
        # Calculating the cost of every epoch
        c = sess.run(cost , feed_dict = {X : x, Y : y})
        print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(W), "b =", 
            sess.run(b))
    # Storing necessary values to be used outside the Session
    training_cost = sess.run(cost , feed_dict ={X: x, Y: y})
    # executing the graph with the value of W
    weight = sess.run(W)
    # executing the graph with the value of b
    bias = sess.run(b)

predictions = weight * x + bias
print("Training cost =", training_cost , "Weight =", weight , "bias =", 
    bias , "\n")

# Plotting the Results

# ro -> red circle with a label of Original data
plt.plot(x, y, "ro", label ="Original data")
plt.plot(x, predictions , label ="Fitted line")
plt.title("Linear Regression Result")
# displays small box containing description of the graph elements 
# for example red dot denotes Original data
plt.legend ()
plt.show()

