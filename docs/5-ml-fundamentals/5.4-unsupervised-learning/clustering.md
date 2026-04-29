---
reading_minutes: 20
objectives:
  - "Distinguish K-means, hierarchical, and DBSCAN by the kinds of data and questions each suits — known k vs unknown, spherical vs arbitrary shape, with vs without noise."
  - "Preprocess for distance-based clustering: handle NaNs and standardise so no single feature dominates Euclidean distance."
  - "Pick a sensible number of clusters using the elbow method on inertia, then sanity-check with silhouette score or domain knowledge."
  - "Avoid the everyday traps: comparing raw cluster-label values across runs, forgetting to scale, and treating clusters as ground-truth classes."
---

# Clustering: Finding Natural Groups in Data

**After this lesson:** you can explain the core ideas in “Clustering: Finding Natural Groups in Data” and reproduce the examples here in your own notebook or environment.

## Overview

Hub for **clustering** ideas: choosing $k$, distances, and validation without labels.

## What is Clustering?

Clustering is like having a smart assistant who can look at a pile of items and automatically organize them into meaningful groups. It's particularly useful when:

- You don't know what groups exist in your data
- You want to discover natural patterns
- You need to segment your data into meaningful categories

## Why Do We Need Clustering?

1. **Customer Segmentation**: Like grouping customers based on their shopping habits
2. **Image Organization**: Like automatically sorting photos by content
3. **Document Clustering**: Like organizing articles by topic
4. **Anomaly Detection**: Like finding unusual patterns in data

## Types of Clustering Algorithms

{% include mermaid-diagram.html src="5-ml-fundamentals/5.4-unsupervised-learning/diagrams/clustering-1.mmd" %}

### 1. K-Means Clustering

Think of K-Means as a smart organizer who:

- Decides how many groups to make (k)
- Places items in the group they're closest to
- Keeps adjusting until everything is in the right place

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Create sample data
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Create visualization
plt.figure(figsize=(10, 5))

# Original data
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# K-Means clusters
plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.savefig('assets/kmeans_example.png')
plt.close()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-12" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Data and K-Means Fit</span>
    </div>
    <div class="code-callout__body">
      <p>Generate four synthetic Gaussian blobs, then fit <code>KMeans(n_clusters=4)</code>; <code>predict</code> assigns each point to the nearest centroid, producing integer cluster labels.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="14-36" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Side-by-side Comparison</span>
    </div>
    <div class="code-callout__body">
      <p>Left subplot shows true labels from <code>make_blobs</code>; right shows K-Means assignments with red X markers at <code>cluster_centers_</code> — comparing the two reveals how well the algorithm recovered the true groups.</p>
    </div>
  </div>
</aside>
</div>

### 2. Hierarchical Clustering

Think of Hierarchical Clustering as building a family tree of your data:

- Starts with each item as its own group
- Gradually combines similar groups
- Creates a tree-like structure of relationships

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Apply Hierarchical Clustering
model = AgglomerativeClustering(n_clusters=4)
y_hc = model.fit_predict(X)

# Create visualization
plt.figure(figsize=(15, 5))

# Original data
plt.subplot(131)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Hierarchical clusters
plt.subplot(132)
plt.scatter(X[:, 0], X[:, 1], c=y_hc, cmap='viridis')
plt.title('Hierarchical Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Dendrogram
plt.subplot(133)
linkage_matrix = linkage(X, method='ward')
dendrogram(linkage_matrix)
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')

plt.tight_layout()
plt.savefig('assets/hierarchical_clustering.png')
plt.close()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-6" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Agglomerative Clustering</span>
    </div>
    <div class="code-callout__body">
      <p>Fit <code>AgglomerativeClustering(n_clusters=4)</code> on the same blob data as K-Means; <code>fit_predict</code> returns integer labels assigned by bottom-up merging.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="8-29" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Three-panel Figure</span>
    </div>
    <div class="code-callout__body">
      <p>Left: original labels; middle: hierarchical assignments; right: <code>dendrogram</code> from scipy's <code>linkage</code> (Ward method) — the dendrogram's branch heights show at what distances clusters merged.</p>
    </div>
  </div>
</aside>
</div>

### 3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

Think of DBSCAN as a smart city planner who:

- Identifies dense neighborhoods (clusters)
- Marks sparse areas as noise
- Doesn't need to know how many neighborhoods to look for

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.cluster import DBSCAN

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_dbscan = dbscan.fit_predict(X)

# Create visualization
plt.figure(figsize=(10, 5))

# Original data
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# DBSCAN clusters
plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.savefig('assets/dbscan_example.png')
plt.close()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-5" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">DBSCAN Parameters</span>
    </div>
    <div class="code-callout__body">
      <p><code>eps=0.3</code> sets the neighborhood radius; <code>min_samples=5</code> sets the density threshold; points labeled -1 by <code>fit_predict</code> are noise (outliers not in any cluster).</p>
    </div>
  </div>
  <div class="code-callout" data-lines="7-26" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Visualize Results</span>
    </div>
    <div class="code-callout__body">
      <p>Side-by-side comparison with true blob labels; DBSCAN may find different cluster boundaries or label some points as noise (-1), shown as a distinct color in the right subplot.</p>
    </div>
  </div>
</aside>
</div>

## How to Choose the Right Algorithm

### Use K-Means when

- You know how many clusters you want
- Your clusters are roughly spherical
- You have a large dataset

### Use Hierarchical Clustering when

- You don't know how many clusters you want
- You want to see the relationships between clusters
- You have a small to medium dataset

### Use DBSCAN when

- You don't know how many clusters you want
- Your clusters can be any shape
- You want to identify outliers

## Best Practices

1. **Data Preprocessing**:

```python
def preprocess_for_clustering(X):
    # Remove missing values
    X = np.nan_to_num(X)
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled
```

2. **Finding the Right Number of Clusters**:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def find_optimal_clusters(X, max_clusters=10):
    # Calculate inertia for different numbers of clusters
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, max_clusters + 1), inertias, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.grid(True)
    plt.savefig('assets/elbow_method.png')
    plt.close()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Inertia Sweep</span>
    </div>
    <div class="code-callout__body">
      <p>Fit K-Means for each k from 1 to <code>max_clusters</code> and collect <code>inertia_</code> (sum of squared distances to centroids); inertia always decreases as k grows but the rate of decrease slows past the true cluster count.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="10-17" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Elbow Plot</span>
    </div>
    <div class="code-callout__body">
      <p>Plot inertia vs k; the "elbow" — where the curve bends sharply — is the heuristic choice for the optimal number of clusters.</p>
    </div>
  </div>
</aside>
</div>

## Common Mistakes to Avoid

1. **Not Scaling Data**: Always standardize your data first
2. **Choosing Wrong Number of Clusters**: Use methods like the elbow method
3. **Using Wrong Algorithm**: Consider your data's characteristics
4. **Ignoring Outliers**: Some algorithms are sensitive to outliers

## Gotchas

- **Cluster labels are arbitrary integers** — running K-Means twice with different random seeds can produce the same clusters but with swapped label numbers (e.g., cluster 0 and cluster 2 swap). Never compare raw label values across runs; use metrics like silhouette score instead.
- **The elbow method is subjective and sometimes has no clear elbow** — on real-world data the inertia curve often decreases smoothly without a visible kink. Pair it with silhouette scores or domain knowledge rather than relying on it alone.
- **Forgetting to scale before clustering** — Euclidean distance is scale-sensitive; a feature in thousands will dominate a feature measured in units, and K-Means/DBSCAN will cluster on that dominant feature almost exclusively.
- **DBSCAN's `eps` is not unitless** — its meaning depends entirely on the scale of your features, so after standardization the same `eps=0.5` behaves very differently than on raw data. Always tune `eps` on scaled data, not the raw values.
- **AgglomerativeClustering can't predict new points** — unlike K-Means, `AgglomerativeClustering` has no `predict` method; you must refit on the combined old + new data to assign labels to unseen points.
- **Assuming clusters found equal ground-truth classes** — clustering is unsupervised, so a "4-cluster" result on the Iris dataset doesn't map cleanly to 3 true species. Validate with adjusted rand index if labels are available, or with domain expertise if they're not.

## Further Reading

1. [Scikit-learn Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html)
2. [Understanding K-Means Clustering](https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1)
3. [DBSCAN Algorithm Explained](https://towardsdatascience.com/dbscan-algorithm-explained-13e3f82f62c6)

## Practice Exercise

Try clustering the famous Iris dataset:

1. Load the data
2. Preprocess it
3. Try different clustering algorithms
4. Compare the results
5. Visualize the clusters

Remember: The goal is to find meaningful patterns in your data!
