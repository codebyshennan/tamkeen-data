---
reading_minutes: 5
objectives:
  - "Walk through Lloyd's algorithm: assign each point to the nearest centroid, recompute centroids, repeat until labels stabilise."
  - "State when k-means is a reasonable choice (roughly spherical clusters of similar size, k known in advance) and when it isn't."
  - "Use `KMeans(n_clusters=k, random_state=...)` and read `labels_`, `cluster_centers_`, and `inertia_` from the fitted estimator."
  - "Recognise common pitfalls: random-seed sensitivity, unscaled features dominating distance, and inertia decreasing monotonically with k."
---

# K-means Clustering

**After this lesson:** you can explain the core ideas in “K-means Clustering” and reproduce the examples here in your own notebook or environment.

## Overview

**K-means**: Lloyd's algorithm, centroids, when spherical clusters are a reasonable assumption, and common failure modes.

## Helpful video

StatQuest overview of K-means clustering.

<iframe width="560" height="315" src="https://www.youtube.com/embed/4b5d3muPQmA" title="K-means Clustering, Clearly Explained" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Quick Reference

{% include mermaid-diagram.html src="5-ml-fundamentals/5.4-unsupervised-learning/diagrams/k-means-clustering-1.mmd" %}

K-means is ideal when:
- You know the approximate number of clusters
- Clusters are roughly spherical
- Clusters have similar sizes

```python
from sklearn.cluster import KMeans

# Basic usage
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
```

For the complete tutorial, see [Clustering Guide](clustering.md).

## Gotchas

- **Random initialization can produce poor local minima** — K-Means converges to the nearest local optimum; two runs with different random seeds can give very different cluster assignments. Set `n_init=10` explicitly to run multiple restarts and keep the best inertia — current scikit-learn defaults to `n_init='auto'`, which runs only **1** restart with the default `k-means++` init, so don't rely on the old "10 by default" behaviour.
- **K-Means assumes spherical, equally-sized clusters** — if your real clusters are elongated, ring-shaped, or very different in size, K-Means will split or merge them incorrectly. Always visualize the result and consider DBSCAN or GMM for non-spherical data.
- **`fit_predict` vs `predict`** — on the *same* data, `fit_predict(X)` (or reading `labels_`) and `predict(X)` return identical labels, because both assign each point to its nearest final centroid. Use `predict` only to assign *new* points to the already-learned centroids without refitting.
- **Inertia always decreases with more clusters** — K=n (one cluster per point) gives inertia=0, which is useless. The elbow method helps, but if there's no clear elbow, silhouette score or domain constraints should guide the final k choice.
- **Not scaling before K-Means** — a feature with large magnitude will dominate the Euclidean distance calculation, causing K-Means to effectively ignore smaller-magnitude features entirely.
