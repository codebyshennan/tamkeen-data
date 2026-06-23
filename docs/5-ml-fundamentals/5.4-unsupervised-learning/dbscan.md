---
reading_minutes: 11
objectives:
  - "Describe DBSCAN's core / border / noise classification and how `eps` and `min_samples` shape the result."
  - "Identify when density clustering beats k-means: arbitrary cluster shapes, unknown cluster count, and explicit outlier handling."
  - "Run `DBSCAN(eps=..., min_samples=...).fit_predict(X)` on scaled data and interpret the `-1` noise label."
  - "Diagnose the two common failure modes: everything-noise and one-giant-cluster, using an `eps` sweep or k-distance plot."
---

# DBSCAN (Density-Based Spatial Clustering)

**After this lesson:** you can fit DBSCAN, visualise its noise labels, and tune `eps` with a practical debugging workflow.

## Overview

DBSCAN finds dense neighborhoods in feature space. Unlike K-Means, it does not ask for the number of clusters in advance. Instead, it asks:

- How close must points be to count as neighbors? (`eps`)
- How many neighbors make an area dense enough? (`min_samples`)

Points in dense areas become clusters. Points that are too isolated are labeled `-1`, meaning noise or outlier.

## Quick Reference

{% include mermaid-diagram.html src="5-ml-fundamentals/5.4-unsupervised-learning/diagrams/dbscan-1.mmd" %}

DBSCAN is ideal when:

- Clusters have arbitrary shapes rather than round blobs.
- You need to identify noise or outliers.
- You do not know the number of clusters.
- The dataset is scaled and distance is meaningful.

## Worked Example

DBSCAN often shines on shapes that confuse centroid-based clustering. The two-moons dataset below has curved clusters, so "nearest centroid" is the wrong mental model.

```python
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

X, true_labels = make_moons(n_samples=300, noise=0.06, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

dbscan = DBSCAN(eps=0.25, min_samples=5)
predicted_labels = dbscan.fit_predict(X_scaled)

plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=true_labels, cmap="viridis")
plt.title("Original moon-shaped groups")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap="viridis")
plt.title("DBSCAN result")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.savefig("assets/dbscan_example.png")
plt.close()
```

<figure>
<img src="assets/dbscan_example.png" alt="Side-by-side scatter plots of moon-shaped data and DBSCAN cluster labels" />
<figcaption>Figure 1: DBSCAN separates curved groups because it follows density instead of centroid distance.</figcaption>
</figure>

## Interpreting Labels

DBSCAN labels are still arbitrary integers, but `-1` has a special meaning:

```python
cluster_ids = set(predicted_labels)
noise_fraction = (predicted_labels == -1).mean()

print("Cluster ids:", cluster_ids)
print("Noise fraction:", round(noise_fraction, 3))
```

If almost everything is `-1`, `eps` is probably too small or `min_samples` is too high. If almost everything is one label, `eps` is probably too large.

## Tuning `eps`

The fastest beginner debugging move is to try a few `eps` values and visualise the result.

```python
eps_values = [0.12, 0.25, 0.50]

fig, axes = plt.subplots(1, len(eps_values), figsize=(15, 4))
for ax, eps in zip(axes, eps_values):
    labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X_scaled)
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
    ax.set_title(f"eps={eps}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

plt.tight_layout()
plt.savefig("assets/dbscan_eps_sweep.png")
plt.close()
```

<figure>
<img src="assets/dbscan_eps_sweep.png" alt="Three DBSCAN plots showing too-small, reasonable, and too-large eps values" />
<figcaption>Figure 2: Small `eps` creates too much noise; large `eps` merges groups. The middle setting preserves the two curved clusters.</figcaption>
</figure>

For a more systematic choice, plot each point's distance to its `min_samples`-th nearest neighbor and look for the elbow. Use that distance as a starting point for `eps`, then inspect the clusters visually.

## Gotchas

- **Noise points labeled `-1` need special handling** - some metrics, including silhouette score, should be computed after filtering noise points (`labels != -1`) and reported alongside the noise fraction.
- **`eps` in raw feature space is meaningless** - DBSCAN uses Euclidean distance, so an `eps` value that works on standardized data can be wildly wrong on unscaled data.
- **All points classified as noise** - if `eps` is too small or `min_samples` is too large, DBSCAN assigns everything to noise. Increase `eps` or reduce `min_samples`.
- **All points in one cluster** - if `eps` is too large, DBSCAN merges everything into one dense region. Decrease `eps`.
- **Different densities are hard** - DBSCAN uses one global `eps`, so dense and sparse clusters can be hard to recover together. Consider HDBSCAN when density varies.

For a longer comparison with K-Means and hierarchical clustering, see the [Clustering Guide](clustering.md).
