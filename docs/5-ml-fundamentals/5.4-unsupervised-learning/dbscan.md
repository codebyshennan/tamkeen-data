# DBSCAN (Density-Based Spatial Clustering)

> **Note**: This content has been consolidated into the advanced clustering guide for a better learning experience.

DBSCAN is covered comprehensively in:

**[Advanced Clustering Techniques](advanced-clustering.md)**

This guide includes:
- How density-based clustering works
- Core points, border points, and noise
- Choosing epsilon and min_samples parameters
- HDBSCAN (Hierarchical DBSCAN)
- Practical examples and visualizations

---

## Quick Reference

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
