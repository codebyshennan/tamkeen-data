---
reading_minutes: 6
objectives:
  - "Describe DBSCAN's core / border / noise classification and how `eps` and `min_samples` shape the result."
  - "Identify when density clustering beats k-means: arbitrary cluster shapes, varying cluster counts, and explicit outlier handling."
  - "Run `DBSCAN(eps=..., min_samples=...).fit_predict(X)` on scaled data and interpret the `-1` noise label."
  - "Diagnose the two common failure modes — everything-noise and one-giant-cluster — using a k-distance plot to set `eps`."
---

# DBSCAN (Density-Based Spatial Clustering)

**After this lesson:** you can explain the core ideas in “DBSCAN (Density-Based Spatial Clustering)” and reproduce the examples here in your own notebook or environment.

## Overview

**DBSCAN** for arbitrary shapes and noise points: `eps`, `min_samples`, and when density clustering wins over k-means.


## Quick Reference

{% include mermaid-diagram.html src="5-ml-fundamentals/5.4-unsupervised-learning/diagrams/dbscan-1.mmd" %}

DBSCAN is ideal when:
- Clusters have arbitrary shapes (not spherical)
- You need to identify noise/outliers
- Cluster sizes and densities vary
- You don't know the number of clusters

```python
from sklearn.cluster import DBSCAN

# Basic usage
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# Noise points are labeled as -1
noise_mask = labels == -1
```

For the complete tutorial, see [Advanced Clustering Guide](advanced-clustering.md).

## Gotchas

- **Noise points labeled `-1` will break silhouette scoring** — `silhouette_score` raises an error if any label is `-1`; filter out noise points (`labels != -1`) before computing it, and report the noise fraction separately.
- **`eps` in raw feature space is meaningless** — DBSCAN uses Euclidean distance, so an `eps` value that works on standardized data will be wildly wrong on unscaled data. Always scale first, then tune `eps`.
- **All points classified as noise** — if `eps` is too small or `min_samples` is too large, DBSCAN assigns everything to noise (-1) with zero clusters. Use a k-distance plot (sort the k-th nearest-neighbor distance for each point) to pick a sensible `eps` before guessing.
- **All points in one cluster** — the opposite failure: if `eps` is too large, DBSCAN merges everything into a single cluster. The k-distance elbow helps here too.
- **DBSCAN struggles with clusters of very different densities** — it uses a single global `eps`, so dense and sparse regions get treated identically; consider HDBSCAN for datasets where cluster density varies significantly.
