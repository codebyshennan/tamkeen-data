# Hierarchical Clustering

> **Note**: This content has been consolidated into the main clustering guide for a better learning experience.

Hierarchical clustering is covered comprehensively in:

**[Clustering: Finding Natural Groups in Data](clustering.md)**

This guide includes:
- Agglomerative vs divisive approaches
- Dendrograms and how to interpret them
- Linkage methods (single, complete, average, ward)
- Cutting dendrograms to form clusters
- Practical examples and visualizations

---

## Quick Reference

Hierarchical clustering is ideal when:
- You want to explore multiple cluster levels
- The natural cluster hierarchy matters
- You don't know the number of clusters in advance

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Basic usage
clustering = AgglomerativeClustering(n_clusters=3)
labels = clustering.fit_predict(X)

# For dendrogram
Z = linkage(X, method='ward')
dendrogram(Z)
```

For the complete tutorial, see [Clustering Guide](clustering.md).
