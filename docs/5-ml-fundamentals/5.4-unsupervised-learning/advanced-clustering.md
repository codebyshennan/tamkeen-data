# Advanced Clustering Techniques

Sometimes basic clustering algorithms aren't enough - we need more sophisticated methods to handle complex data patterns. Let's explore advanced clustering techniques that can tackle challenging scenarios!

## HDBSCAN: Advanced Density-Based Clustering

HDBSCAN improves on DBSCAN by:

- Automatically adapting to varying densities
- Not requiring an epsilon parameter
- Providing cluster membership probabilities

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
import hdbscan

# Create complex dataset
X1, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
X2, _ = make_blobs(n_samples=100, centers=[[2, 0]], cluster_std=0.5, random_state=42)
X = np.vstack([X1, X2])

# Apply HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
cluster_labels = clusterer.fit_predict(X)

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
plt.title('HDBSCAN Clustering')
plt.colorbar(scatter)
plt.show()

# Plot cluster probabilities
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X[:, 0], X[:, 1], 
                     c=clusterer.probabilities_, cmap='viridis')
plt.title('HDBSCAN Cluster Membership Probabilities')
plt.colorbar(scatter)
plt.show()
```

## Gaussian Mixture Models (GMM)

GMM is like having multiple overlapping probability distributions:

1. Each cluster is a Gaussian distribution
2. Points can belong partially to multiple clusters
3. Model learns distribution parameters

```python
from sklearn.mixture import GaussianMixture

# Create sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=[1.0, 2.0, 0.5, 1.5],
                  random_state=42)

# Fit GMM
gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(X)
labels = gmm.predict(X)
probs = gmm.predict_proba(X)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot cluster assignments
scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
ax1.set_title('GMM Cluster Assignments')
plt.colorbar(scatter1, ax=ax1)

# Plot membership probabilities for first cluster
scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=probs[:, 0], cmap='viridis')
ax2.set_title('Probability of Cluster 1 Membership')
plt.colorbar(scatter2, ax=ax2)

plt.tight_layout()
plt.show()
```

## Spectral Clustering

Spectral clustering is like finding communities in a social network:

1. Build similarity graph
2. Find graph Laplacian
3. Use eigenvectors for clustering

```python
from sklearn.cluster import SpectralClustering

# Create interlocking circles
from sklearn.datasets import make_circles
X, _ = make_circles(n_samples=1000, factor=0.5, noise=0.05, random_state=42)

# Apply Spectral Clustering
spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                             random_state=42)
labels = spectral.fit_predict(X)

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('Spectral Clustering')
plt.colorbar(scatter)
plt.show()
```

## Real-World Applications

### 1. Topic Modeling with GMM

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "machine learning algorithms classification",
    "neural networks deep learning",
    "clustering unsupervised learning",
    "deep neural networks training",
    "kmeans clustering algorithm"
]

# Create TF-IDF features
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(documents).toarray()

# Apply GMM
gmm = GaussianMixture(n_components=2, random_state=42)
doc_labels = gmm.fit_predict(X)
doc_probs = gmm.predict_proba(X)

# Print results
for doc, label, probs in zip(documents, doc_labels, doc_probs):
    print(f"Document: {doc}")
    print(f"Topic: {label}")
    print(f"Topic Probabilities: {probs}\n")
```

### 2. Image Segmentation with HDBSCAN

```python
from skimage import io
from skimage.color import rgb2lab

# Load and prepare image
image = io.imread('sample_image.jpg')
pixels = rgb2lab(image).reshape(-1, 3)

# Apply HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
labels = clusterer.fit_predict(pixels)

# Reshape and display results
segmented = labels.reshape(image.shape[:2])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(image)
ax1.set_title('Original Image')
ax2.imshow(segmented, cmap='viridis')
ax2.set_title('HDBSCAN Segmentation')
plt.show()
```

## Advanced Techniques

### 1. Ensemble Clustering

```python
def ensemble_clustering(X, n_members=5):
    # Create ensemble members
    clusterers = [
        hdbscan.HDBSCAN(min_cluster_size=5),
        GaussianMixture(n_components=3),
        SpectralClustering(n_clusters=3),
    ]
    
    # Get predictions from each member
    predictions = np.zeros((X.shape[0], len(clusterers)))
    for i, clusterer in enumerate(clusterers):
        predictions[:, i] = clusterer.fit_predict(X)
    
    # Combine predictions (simple majority voting)
    from scipy.stats import mode
    ensemble_pred = mode(predictions, axis=1)[0]
    
    return ensemble_pred
```

### 2. Semi-Supervised Clustering

```python
def semi_supervised_gmm(X, labeled_indices, true_labels):
    # Initialize GMM
    gmm = GaussianMixture(n_components=len(np.unique(true_labels)))
    
    # Partial fit with labeled data
    X_labeled = X[labeled_indices]
    gmm.fit(X_labeled, true_labels[labeled_indices])
    
    # Predict remaining points
    labels = gmm.predict(X)
    
    return labels
```

### 3. Online Clustering

```python
from sklearn.cluster import MiniBatchKMeans

def online_clustering(data_generator, n_clusters=3):
    # Initialize online clusterer
    clusterer = MiniBatchKMeans(n_clusters=n_clusters)
    
    # Process data in batches
    for batch in data_generator:
        clusterer.partial_fit(batch)
    
    return clusterer
```

## Best Practices

### 1. Model Selection

```python
def select_best_model(X, models, n_splits=5):
    from sklearn.metrics import silhouette_score
    scores = {}
    
    for name, model in models.items():
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[name] = score
    
    return scores
```

### 2. Parameter Optimization

```python
def optimize_hdbscan(X):
    best_score = -1
    best_params = {}
    
    for min_cluster_size in [5, 10, 15, 20]:
        for min_samples in [5, 10, 15]:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples
            )
            labels = clusterer.fit_predict(X)
            
            if len(np.unique(labels)) > 1:  # More than one cluster
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_params = {
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples
                    }
    
    return best_params
```

## Common Pitfalls and Solutions

1. **Model Selection Issues**
   - Try multiple algorithms
   - Use ensemble methods
   - Validate results

2. **Parameter Sensitivity**
   - Use parameter search
   - Cross-validate results
   - Consider stability

3. **Scalability**
   - Use mini-batch methods
   - Consider data sampling
   - Implement parallel processing

## Next Steps

Now that you've mastered clustering techniques, try the [assignment](./assignment.md) to apply these concepts to real-world problems!
