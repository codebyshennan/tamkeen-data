# Clustering Algorithms

Imagine sorting a deck of cards into suits - you naturally group similar cards together. Clustering algorithms do the same thing with data, finding natural groups based on similarity! Let's explore these fascinating techniques. 🃏

## Understanding Clustering 🎯

Clustering helps us:
1. Discover natural groups in data
2. Find patterns and relationships
3. Reduce data complexity
4. Generate insights

## K-means Clustering 📊

K-means is like finding the centers of crowds in a plaza:
1. Pick K random points as centers
2. Assign each point to nearest center
3. Move centers to middle of their groups
4. Repeat until stable

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Create sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Apply K-means
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X)

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', 
           s=200, linewidth=3, label='Centers')
plt.title('K-means Clustering')
plt.legend()
plt.show()
```

### Finding Optimal K
```python
def plot_elbow_curve(X, max_k=10):
    inertias = []
    K = range(1, max_k+1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(K, inertias, 'bo-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()

plot_elbow_curve(X)
```

## DBSCAN 🌟

DBSCAN is like finding groups of friends at a party:
1. Start with one person
2. Find all friends within arm's reach
3. Find their friends
4. Repeat until no more connected friends

```python
from sklearn.cluster import DBSCAN
import numpy as np

# Create sample data with noise
X_moons, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
clusters = dbscan.fit_predict(X_moons)

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_moons[:, 0], X_moons[:, 1], 
                     c=clusters, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.colorbar(scatter)
plt.show()
```

### Parameter Selection
```python
def explore_dbscan_parameters(X, eps_range=[0.1, 0.3, 0.5], 
                            min_samples_range=[3, 5, 10]):
    fig, axes = plt.subplots(len(eps_range), len(min_samples_range), 
                            figsize=(15, 15))
    
    for i, eps in enumerate(eps_range):
        for j, min_samples in enumerate(min_samples_range):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(X)
            
            axes[i, j].scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
            axes[i, j].set_title(f'eps={eps}, min_samples={min_samples}')
    
    plt.tight_layout()
    plt.show()

explore_dbscan_parameters(X_moons)
```

## Hierarchical Clustering 🌳

Like creating a family tree of data points:
1. Start with each point as its own cluster
2. Merge closest clusters
3. Repeat until one cluster remains

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Create sample data
X, _ = make_blobs(n_samples=50, centers=3, random_state=42)

# Create linkage matrix
linkage_matrix = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Apply clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
clusters = hierarchical.fit_predict(X)

# Plot clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.title('Hierarchical Clustering Results')
plt.colorbar(scatter)
plt.show()
```

## Real-World Applications 🌟

### 1. Customer Segmentation
```python
# Create customer data
np.random.seed(42)
n_customers = 500

# Generate features
recency = np.random.exponential(50, n_customers)
frequency = np.random.normal(10, 5, n_customers)
monetary = np.random.normal(100, 50, n_customers)

# Combine features
X_customers = np.column_stack([recency, frequency, monetary])

# Scale features
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X_customers)

# Apply K-means
kmeans = KMeans(n_clusters=4, random_state=42)
segments = kmeans.fit_predict(X_scaled)

# Visualize results
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2],
                    c=segments, cmap='viridis')
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
plt.colorbar(scatter)
plt.title('Customer Segments')
plt.show()
```

### 2. Image Segmentation
```python
from sklearn.cluster import MiniBatchKMeans
from skimage import io

# Load and reshape image
image = io.imread('sample_image.jpg')
pixels = image.reshape(-1, 3)

# Apply clustering
n_colors = 5
kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42)
labels = kmeans.fit_predict(pixels)

# Create segmented image
segmented_image = kmeans.cluster_centers_[labels].reshape(image.shape)

# Display results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(image)
ax1.set_title('Original Image')
ax2.imshow(segmented_image.astype('uint8'))
ax2.set_title('Segmented Image')
plt.show()
```

## Best Practices 🎯

### 1. Data Preprocessing
```python
def preprocess_for_clustering(X):
    # Handle missing values
    X = np.nan_to_num(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Remove outliers if needed
    from scipy import stats
    z_scores = stats.zscore(X_scaled)
    X_clean = X_scaled[np.all(np.abs(z_scores) < 3, axis=1)]
    
    return X_clean
```

### 2. Cluster Validation
```python
from sklearn.metrics import silhouette_score

def validate_clustering(X, labels):
    # Silhouette score
    silhouette = silhouette_score(X, labels)
    
    # Davies-Bouldin score
    davies = davies_bouldin_score(X, labels)
    
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Davies-Bouldin Score: {davies:.3f}")
```

### 3. Visualization
```python
def plot_clusters_2d(X, labels, centers=None):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], 
                   c='red', marker='x', s=200, linewidth=3)
    
    plt.colorbar(scatter)
    plt.title('Clustering Results')
    plt.show()
```

## Common Pitfalls and Solutions 🚧

1. **K-means Issues**
   - Sensitive to initialization
   - Assumes spherical clusters
   - Requires knowing K
   
   Solutions:
   - Use k-means++
   - Try multiple initializations
   - Use elbow method/silhouette analysis

2. **DBSCAN Challenges**
   - Sensitive to parameters
   - Struggles with varying densities
   - Memory intensive
   
   Solutions:
   - Use parameter search
   - Consider HDBSCAN
   - Use data sampling

3. **Hierarchical Clustering Limitations**
   - Computationally expensive
   - Memory intensive
   - Can't undo merges
   
   Solutions:
   - Use smaller datasets
   - Consider mini-batch methods
   - Try different linkage methods

## Next Steps

Now that you understand clustering algorithms, let's explore [Advanced Clustering](./advanced-clustering.md) techniques like HDBSCAN and Gaussian Mixture Models!
