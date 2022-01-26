# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 21:38:27 2022

@author: kkakh
"""

# Hierarchicsl clustering
# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dedrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# train the Hierarchical clustering model to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

print(y_hc)

# Visualize the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c= 'red', label= 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c= 'blue', label= 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c= 'green', label= 'Cluster 3')
plt.title('CLusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
