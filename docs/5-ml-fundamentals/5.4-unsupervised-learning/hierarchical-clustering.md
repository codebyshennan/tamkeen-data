---
reading_minutes: 6
objectives:
  - "Explain bottom-up agglomerative clustering and how a dendrogram encodes the order in which clusters merge."
  - "Compare ward / complete / average / single linkage and how each shapes the resulting cluster geometry."
  - "Use `AgglomerativeClustering` and `scipy.cluster.hierarchy.linkage` + `dendrogram` to fit, plot, and pick a cut height."
  - "Recognise practical limits: O(n²) memory, no `predict` for new points, and why ward linkage requires Euclidean distance."
---

# Hierarchical Clustering

**After this lesson:** you can explain the core ideas in “Hierarchical Clustering” and reproduce the examples here in your own notebook or environment.

## Overview

**Agglomerative** clustering: linkage criteria, dendrograms, and choosing a cut.


## Quick Reference

{% include mermaid-diagram.html src="5-ml-fundamentals/5.4-unsupervised-learning/diagrams/hierarchical-clustering-1.mmd" %}

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

## Gotchas

- **Ward linkage requires Euclidean distance** — `method='ward'` in `scipy.cluster.hierarchy.linkage` only works with Euclidean distance; passing a precomputed distance matrix or using it with Manhattan distance will produce incorrect or error-prone results.
- **Cutting the dendrogram at the wrong height** — the horizontal cut on a dendrogram determines the number of clusters, but learners often cut at an aesthetically pleasing point rather than the largest vertical gap; look for the longest vertical line with no horizontal cuts crossing it.
- **Scalability wall** — agglomerative clustering is O(n²) in memory and O(n² log n) in time; on datasets larger than ~10,000 points it becomes impractically slow and scikit-learn's `AgglomerativeClustering` may raise a memory error without warning.
- **No `predict` for new points** — `AgglomerativeClustering` has no `predict` method, unlike K-Means; if you need to assign new data points to existing clusters you must re-run the full fit on the combined dataset.
- **Linkage choice silently changes cluster shapes** — `single` linkage tends to produce long chained clusters, `complete` tends toward compact ones, and `average` is a compromise. Swapping linkage without re-inspecting the dendrogram can produce completely different groupings on the same data.
