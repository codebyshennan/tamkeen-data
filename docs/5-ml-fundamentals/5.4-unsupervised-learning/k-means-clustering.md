---
reading_minutes: 10
objectives:
  - "Walk through Lloyd's algorithm: assign each point to the nearest centroid, recompute centroids, repeat until labels stabilise."
  - "State when k-means is a reasonable choice: roughly spherical clusters of similar size, with `k` known in advance."
  - "Use `KMeans(n_clusters=k, random_state=...)` and inspect `labels_`, `cluster_centers_`, and `inertia_` from the fitted estimator."
  - "Recognise common pitfalls: random-seed sensitivity, unscaled features dominating distance, and inertia decreasing monotonically with `k`."
---

# K-means Clustering

**After this lesson:** you can fit K-Means, visualise the cluster assignments, and explain what the centroids and inertia mean.

## Overview

K-Means is a centroid-based clustering algorithm. It tries to divide points into `k` groups by repeatedly answering two questions:

1. Which centroid is each point closest to?
2. Where should each centroid move after seeing its assigned points?

It works best when the groups are compact, round-ish, and similar in size. It is a poor fit for long curved shapes, heavy outliers, or datasets where you do not have a reasonable guess for `k`.

## Helpful video

StatQuest overview of K-means clustering.

<iframe width="560" height="315" src="https://www.youtube.com/embed/4b5d3muPQmA" title="K-means Clustering, Clearly Explained" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Mental Model

{% include mermaid-diagram.html src="5-ml-fundamentals/5.4-unsupervised-learning/diagrams/k-means-clustering-1.mmd" %}

K-Means alternates between assignment and update steps:

- **Assign:** each point goes to the nearest centroid.
- **Update:** each centroid moves to the mean of the points assigned to it.
- **Repeat:** stop when assignments stop changing or the maximum iterations are reached.

The final cluster labels are arbitrary integers. Cluster `0` does not mean "first", "best", or "smallest"; it only names one discovered group.

## Worked Example

This example creates four obvious groups, fits K-Means, and plots the learned clusters beside the true synthetic labels.

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Create a beginner-friendly dataset with four visible groups.
X, true_labels = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=0.60,
    random_state=0,
)

# n_init=10 tries 10 random centroid starts and keeps the best result.
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
predicted_labels = kmeans.fit_predict(X)

plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=true_labels, cmap="viridis")
plt.title("Original synthetic groups")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap="viridis")
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="red",
    marker="x",
    s=200,
    linewidths=3,
    label="Centroids",
)
plt.title("K-Means result")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.tight_layout()
plt.savefig("assets/kmeans_example.png")
plt.close()
```

<figure>
<img src="assets/kmeans_example.png" alt="Side-by-side scatter plots of original synthetic groups and K-Means clusters with centroid markers" />
<figcaption>Figure 1: K-Means recovers compact blob-shaped groups and marks each centroid with a red X.</figcaption>
</figure>

## Reading the Fitted Model

After fitting, the estimator stores useful attributes:

```python
print("Cluster labels:", predicted_labels[:10])
print("Centroids:")
print(kmeans.cluster_centers_)
print("Inertia:", round(kmeans.inertia_, 2))
```

- `labels_` or `fit_predict(X)` gives one cluster id per row.
- `cluster_centers_` gives the learned centroid coordinates.
- `inertia_` is the total squared distance from points to their assigned centroids. Lower inertia is better for a fixed `k`, but it always decreases when you add more clusters.

## Choosing `k`

Start with a domain guess, then compare it with an elbow plot from the full [Clustering Guide](clustering.md). If `k=4` and `k=5` have similar inertia, prefer the simpler result unless domain knowledge says the fifth group matters.

Use K-Means when:

- You can choose a reasonable `k`.
- Features are numeric and scaled.
- You expect compact, roughly spherical clusters.
- You need a fast baseline for many rows.

Avoid K-Means when:

- Clusters are crescent-shaped, ring-shaped, or strongly elongated.
- Outliers are important instead of noise to ignore.
- Cluster density varies a lot across the dataset.

## Gotchas

- **Random initialization can produce poor local minima** - K-Means converges to the nearest local optimum. Set `n_init=10` explicitly to run multiple restarts and keep the best inertia.
- **K-Means assumes spherical, equally-sized clusters** - if your real clusters are elongated, ring-shaped, or very different in size, K-Means will split or merge them incorrectly. Always visualize the result and consider DBSCAN or GMM for non-spherical data.
- **`fit_predict` vs `predict`** - on the same data, `fit_predict(X)` and `predict(X)` return identical labels. Use `predict` only to assign new points to already-learned centroids without refitting.
- **Inertia always decreases with more clusters** - `k=n` gives inertia 0, which is useless. Pair the elbow method with silhouette score or business constraints.
- **Not scaling before K-Means** - a feature with large magnitude will dominate Euclidean distance, causing K-Means to ignore smaller-magnitude features.
