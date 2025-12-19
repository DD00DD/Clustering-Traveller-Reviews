import pandas as pd
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

# Load dataset
data = pd.read_csv(r'C:\Users\Administrator\Downloads\cps803_a4\tripadvisor_review.csv')

# Drop the first column (UserID)
X = data.drop(data.columns[0], axis=1)

# === Hierarchical clustering ===
# Using linkage to train hierarchy model (complete linkage = max)
Z = hierarchy.linkage(X, method='complete')

# Assign cluster labels
k = 3
cluster_labels = hierarchy.fcluster(Z, k, criterion='maxclust')

# Add cluster labels to the original dataframe
data['Cluster'] = cluster_labels

# Plot dendrogram with colored clusters
plt.figure(figsize=(12, 8))
hierarchy.dendrogram(
    Z,
    leaf_rotation=90,
    leaf_font_size=10,
    color_threshold=Z[-(k-1), 2]  # automatically color clusters
)
plt.xlabel("Users")
plt.ylabel("Distance")
plt.show()

# Visualize clusters in 2D using PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, s=50)
plt.xlabel('Y')
plt.ylabel('X')
plt.show()


# === K-means clustering ===
num_clusters = 3  # Perfect, Average, Poor

# Run K-Means on all 10 categories at once
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(X.values)

# containing the cluster index for each data point.
labels = kmeans.labels_

# Compute mean rating of each cluster (average across all 10 categories)
cluster_means = []
for c in range(num_clusters):
    cluster_points = X.values[labels == c]  
    mean_value = cluster_points.mean()  # average across all points and categories
    cluster_means.append((c, mean_value)) # put tuple in means list with the category next to its average

# Sort clusters descending by mean rating to see which cluster per category is the highest
cluster_means_sorted = sorted(cluster_means, key=lambda x: x[1], reverse=True)

# Map clusters to Perfect, Average, or Poor
cluster_mapping = {}
cluster_mapping[cluster_means_sorted[0][0]] = 'Perfect'
cluster_mapping[cluster_means_sorted[1][0]] = 'Average'
cluster_mapping[cluster_means_sorted[2][0]] = 'Poor'

# Apply mapping to all labels classify them in one of three clusters (perfect, average, or poor)
labeled_clusters = np.array([cluster_mapping[l] for l in labels])

# transform the 10 attribute to 2D to plot with PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X.values)

plt.figure(figsize=(8, 6))
for cluster_name in ['Perfect','Average','Poor']:
    mask = labeled_clusters == cluster_name # mask variable is here to help seperate and color each point under their group
    plt.scatter(X_2d[mask,0], X_2d[mask,1], s=50, label=cluster_name)
plt.xlabel('Y')
plt.ylabel('X')
plt.title('K-means Clusters of Destinations Across All Categories')
plt.legend()
plt.show()