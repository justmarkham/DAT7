'''
CLASS: Clustering
'''

# beer dataset
import pandas as pd
url = 'https://raw.githubusercontent.com/justmarkham/DAT7/master/data/beer.txt'
beer = pd.read_csv(url, sep=' ')
beer

# define X
X = beer.drop('name', axis=1)

'''
K-means
'''

# K-means with 3 clusters
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=1)
km.fit(X)

# review the cluster labels (based on calories only, due to scale)
km.labels_
beer['cluster'] = km.labels_
beer.sort('cluster')

# review the cluster centers
km.cluster_centers_
beer.groupby('cluster').mean()
centers = beer.groupby('cluster').mean()

# create colors array for plotting
import numpy as np
colors = np.array(['red', 'green', 'blue'])

# plot clusters with their centers
import matplotlib.pyplot as plt
plt.scatter(beer.calories, beer.alcohol, c=colors[beer.cluster], s=50)
plt.scatter(centers.calories, centers.alcohol, linewidths=3, marker='+', s=300, c='black')
plt.xlabel('calories')
plt.ylabel('alcohol')

# scatter plot matrix
pd.scatter_matrix(X, c=colors[beer.cluster], figsize=(10,10), s=100)

# center and scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means with 3 clusters on scaled data
km.fit(X_scaled)
beer['cluster'] = km.labels_
beer.sort('cluster')
beer.groupby('cluster').mean()

# scatter plot matrix with scaled data
beer_scaled = pd.DataFrame(X_scaled, columns=beer.columns[1:-1])
pd.scatter_matrix(beer_scaled, c=colors[beer.cluster], figsize=(10,10), s=100)

'''
Silhouette Coefficient

SC is calculated for each observation as follows:
a = mean distance to all other points in its cluster
b = mean distance to all other points in the next nearest cluster
SC = (b-a)/max(a, b)

SC ranges from -1 (worst) to 1 (best).

A global SC is calculated by taking the mean of the SC for all observations.
'''

# calculate SC for K=3
from sklearn import metrics
metrics.silhouette_score(X_scaled, km.labels_)

# calculate SC for K=2 through K=19
k_range = range(2, 20)
scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=1)
    km.fit(X_scaled)
    scores.append(metrics.silhouette_score(X_scaled, km.labels_))

# plot the results
plt.plot(k_range, scores)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficient')
plt.grid(True)

# K-means with 4 clusters on scaled data
km = KMeans(n_clusters=4, random_state=1)
km.fit(X_scaled)
beer['cluster'] = km.labels_
beer.sort('cluster')

'''
DBSCAN
'''

# DBSCAN with eps=1 and min_samples=3
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=1, min_samples=3)
db.fit(X_scaled)

# review the cluster labels
db.labels_
beer['cluster'] = db.labels_
beer.sort('cluster')
beer.groupby('cluster').mean()

# scatter plot matrix with scaled data
pd.scatter_matrix(beer_scaled, c=colors[beer.cluster], figsize=(10,10), s=100)
