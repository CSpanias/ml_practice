import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# import dataset
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")

# assing X variables
X = df[['total_bill', 'size']]

# assign algorithm
model = KMeans(n_clusters=6)

# fit algorithm to data
model.fit(X)

# run algorithm
model_predict = model.predict(X)
# centroid coordinates
centroids = model.cluster_centers_
print(centroids)

# plot centroids and clusters
plt.figure(figsize=(7, 5))
# set x- and y-axis, color code according to centroid, color scheme, size
plt.scatter(X['total_bill'], X['size'], c=model_predict, cmap='rainbow', s=50)
# alpha = transparency (superimpose centroids on top of other points)
plt.scatter(centroids[:,0], centroids[:,1], c= 'black', s=200, alpha=1)
plt.show()

