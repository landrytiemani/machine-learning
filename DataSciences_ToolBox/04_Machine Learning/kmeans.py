# Title: K-Means Clustering with Elbow Method (Iris Dataset)

# Purpose: This script demonstrates K-means clustering on the Iris dataset, determining the optimal number of clusters using the Elbow method and visualizing the results.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load and prepare data
iris = load_iris()
X = iris.data

# Standardize the data (important for K-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Elbow Method for Optimal k
inertia = []  # Sum of squared distances to closest centroid
k_values = range(1, 31)  # Test different numbers of clusters

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method results
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.xticks(k_values)
plt.show()

# Choose optimal k (based on the plot, let's say k=3)
optimal_k = 3

# Final K-means with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
y_pred = kmeans.fit_predict(X_scaled)


# Compare clusters to actual labels
print("Comparison of Clusters to Actual Labels:")
print(pd.crosstab(iris.target, y_pred))

# Visualize (if you have only 2 features)
if X_scaled.shape[1] == 2:  # Works only for 2-dimensional data
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
    plt.title(f'K-means Clustering (k={optimal_k})')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.show()

