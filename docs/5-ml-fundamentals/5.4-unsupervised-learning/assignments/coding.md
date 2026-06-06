# Assignment: Unsupervised Learning with K-Means and PCA

> **Submit your work on Skills Union →** <https://skillsu.com/member/assessment>

## Setup

Run the following code to load and prepare the dataset:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load iris dataset (150 samples, 4 features, 3 species)
iris = load_iris()
X = iris.data
feature_names = iris.feature_names
true_labels = iris.target          # used only for visual comparison, NOT as input to clustering

# Scale features — required before both K-Means and PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Dataset shape  : {X.shape}")
print(f"Feature names  : {feature_names}")
print(f"Unique classes : {np.unique(true_labels)}  (3 iris species)")
```

```
Dataset shape  : (150, 4)
Feature names  : ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Unique classes : [0 1 2]  (3 iris species)
```

## Tasks

### 1. K-Means Clustering — Find the Elbow

- Import `KMeans` from `sklearn.cluster`.
- Fit K-Means for each `k` in `range(1, 11)` using `X_scaled` (not `X`).
  Use `KMeans(n_clusters=k, random_state=42, n_init=10)` for reproducibility.
- Record `kmeans.inertia_` (within-cluster sum of squares) for each `k`.
- Plot inertia vs `k` as a line chart. Label the axes and add a title "Elbow Method".
- Visually identify the elbow point and print: `"Chosen k based on elbow: <value>"`.
- Fit a final `KMeans` model with your chosen `k` on `X_scaled` and store `cluster_labels = kmeans.labels_`.

### 2. PCA — Explained Variance

- Import `PCA` from `sklearn.decomposition`.
- Fit `PCA()` (all components) on `X_scaled`.
- Print `explained_variance_ratio_` for each component.
- Plot the cumulative explained variance against the number of components.
  Add a horizontal dashed line at 95% and label the axes and title.
- Print how many components are needed to exceed 95% cumulative explained variance.

### 3. Visualise Clusters After PCA Reduction

- Fit a new `PCA(n_components=2)` on `X_scaled` and transform to get `X_2d` (shape 150 × 2).
- Create a 1×2 subplot figure:
  - **Left:** Scatter `X_2d` coloured by your K-Means `cluster_labels` from Task 1.
    Title: "K-Means Clusters (PCA 2D)".
  - **Right:** Scatter `X_2d` coloured by `true_labels`.
    Title: "True Species Labels (PCA 2D)".
- Add a colour bar or legend to both subplots.
- In a comment, describe how well the K-Means clusters align with the true species labels.

### 4. Interpret Cluster Centroids

- Using the final K-Means model from Task 1, retrieve `kmeans.cluster_centers_` (shape k × 4, in scaled space).
- Inverse-transform the centroids back to original units using `scaler.inverse_transform(...)`.
- Print a table (or formatted output) showing each cluster's centroid values for all four features.
- In a short comment (2–3 lines), describe which features vary most between clusters and what that might mean biologically.

## Deliverable

Submit a single Python script that:

1. Runs all four tasks in order — no external data files required.
2. Produces the three required plots (elbow curve, cumulative variance, cluster vs true label scatter).
3. Prints labelled numeric outputs for inertia values, explained variance ratios, chosen k, and centroid table.
4. Includes brief comments interpreting the results of each task.

## Hints

<details>
<summary>Show hints</summary>

### 1. K-Means — Elbow
- **Where:** [K-Means Clustering](../k-means-clustering.md) — "Gotchas: Inertia always decreases with more clusters"; [Clustering](../clustering.md) — "Elbow method".
- **Think:** Inertia measures how tightly points cluster around their centroid — it always decreases as k grows. The elbow is the point where adding another cluster gives diminishing returns. `KMeans.inertia_` is available after `fit`; collect it in a list before plotting.

### 2. PCA — Explained Variance
- **Where:** [PCA](../pca.md) — "Standardise before fitting PCA, then read `explained_variance_ratio_`".
- **Think:** `pca.explained_variance_ratio_` is an array where each element is the fraction of variance explained by one component. `np.cumsum(...)` converts it to the cumulative version. Fit on `X_scaled` (not raw data) — the lesson explains why standardisation is mandatory before PCA.

### 3. Cluster Visualisation
- **Where:** [PCA](../pca.md) — "Use `fit_transform` for projection"; [Clustering](../clustering.md) — "Why Do We Need Clustering".
- **Think:** Fit `PCA(n_components=2)` on `X_scaled` using `fit_transform` in one step. Use `c=cluster_labels` and `c=true_labels` in `plt.scatter` to colour by different label sources. The comparison tells you whether the algorithm found groups that match biological species — a useful sanity check for unsupervised results.

### 4. Centroid Interpretation
- **Where:** [K-Means Clustering](../k-means-clustering.md) — "`cluster_centers_`"; [Clustering](../clustering.md) — k-Means section.
- **Think:** `kmeans.cluster_centers_` is in scaled space (z-scores). Use `scaler.inverse_transform(kmeans.cluster_centers_)` to recover original units (cm). Compare centroid values across clusters for each feature to find which feature shows the largest spread — that feature is most discriminative.

### Common pitfalls
- Running K-Means on raw (unscaled) `X` instead of `X_scaled` will cause sepal length (~5–7 cm) to dominate over petal width (~0.1–2.5 cm), producing clusters driven almost entirely by one feature.
- Fitting `PCA` on raw `X` instead of `X_scaled` gives principal components dominated by whichever feature has the largest variance — always standardise first.
- Comparing cluster labels numerically across runs is not meaningful — K-Means assigns cluster integers arbitrarily. Cluster "0" in one run may correspond to cluster "2" in another.
- Using `inertia_` as the sole criterion: if there is no clear elbow, supplement with silhouette score (`sklearn.metrics.silhouette_score`) or rely on domain knowledge.

</details>
