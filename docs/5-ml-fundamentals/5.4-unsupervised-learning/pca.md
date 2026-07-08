---
reading_minutes: 22
objectives:
  - >-
    Define principal components as orthogonal directions of maximum variance and
    connect them to eigenvectors of the covariance matrix.
  - >-
    Standardise before fitting `PCA`, then read `explained_variance_ratio_` and
    its cumulative sum to choose `n_components`.
  - >-
    Use `fit_transform` and `inverse_transform` for projection, reconstruction,
    and image-style compression, and treat reconstruction MSE as the lossiness
    measure.
  - >-
    Avoid leakage and over-interpretation: fit PCA on training data only, never
    name a PC after a single feature, and switch to non-linear methods (kernel
    PCA, t-SNE, UMAP) when the manifold is curved.
---

# Principal Component Analysis (PCA): Simplifying Complex Data

**After this lesson:** you can explain Principal Component Analysis (PCA): Simplifying Complex Data and try the examples in your own notebook.

## Overview

**PCA** finds orthogonal **principal components**-directions of maximum variance, and lets you project data onto the top few for visualization, denoising, or as inputs to other models. **Prerequisites:** vectors and eigenvalue intuition from Module 1 linear algebra; [unsupervised learning hub](./).

Imagine you're trying to describe a person to someone who's never met them. Instead of listing every single detail (height, weight, hair color, eye color, clothing, etc.), you might focus on the most distinctive features that make them recognizable. That's exactly what PCA does with data - it helps us focus on the most important aspects while simplifying the rest!

## Helpful video

StatQuest: Principal Component Analysis (PCA), step by step.

## What is PCA?

PCA is like creating a simplified map of a complex city. Just as a map helps you navigate a city by showing the most important streets and landmarks, PCA helps you navigate complex data by showing the most important features.

### Why Do We Need PCA?

1. **Too Many Features**: Imagine trying to understand a person by looking at 100 different measurements. It's overwhelming! PCA helps us focus on the most important ones.
2. **Visualization**: It's hard to visualize data with more than 3 dimensions. PCA helps us see patterns in high-dimensional data by reducing it to 2D or 3D.
3. **Noise Reduction**: Like removing background noise from a recording, PCA helps us focus on the important signals in our data.

## How Does PCA Work?

break it down into simple steps:

1. **Standardize the Data**: First, we make sure all features are on the same scale (like converting different currencies to dollars).
2. **Find Principal Components**: These are like the main directions in which our data varies the most.
3. **Project the Data**: We rotate our data to align with these main directions.

_The output components are **uncorrelated**, PC1 captures the most variance, PC2 the next most, and so on. The scree plot shows where adding more components stops being useful._

Look at this in action with a simple example:

#### 2D toy cloud: scale, fit PCA, three subplots

Scale before PCA

PCA finds the directions of maximum _variance_. A feature measured in thousands will dominate over one measured in units, so `StandardScaler` is applied first to give every feature equal footing.

Fit and transform

`PCA()` with no argument keeps all components. `fit_transform` finds the principal directions _and_ projects the data onto them in one step, equivalent to calling `fit` then `transform`.

Principal component arrows

`pca.components_` holds the principal directions as unit vectors. Drawing them as arrows on the original data shows _which way the data varies most_, the longer the effective spread, the more variance that component captures.

PC space + explained variance

The third subplot shows data in **PC coordinates**: axes are now orthogonal directions of maximum variance. `explained_variance_ratio_` says what fraction of total variance each PC accounts for, two roughly-equal values here confirm neither direction dominates.

```
Explained variance ratio: [0.50565666 0.49434334]
```

## Real-World Example: Image Compression

Look at how PCA can help compress images while maintaining quality:

#### Digits reconstruction vs number of components

Load Digits and Setup

Load the 8×8 digit pixel dataset (64 features); a 2-row subplot grid will show the original image on top and reconstructions from increasing component counts below.

Original Row

Fill the top row with the same original digit image four times as a reference; reshaping the flat 64-value array to (8, 8) is required for `imshow`.

Reconstruction Loop

For each component count, fit PCA on all digits, project, then call `inverse_transform` to reconstruct; the title shows cumulative explained variance so readers see the quality-compression trade-off at a glance.

## How to Choose the Number of Components

### Method 1: Explained Variance Ratio

Think of this like a pie chart showing how much each component contributes to the total information:

#### Cumulative explained variance curve

Full PCA Fit

Fit PCA with no component limit to get all eigenvalues; `np.cumsum` on `explained_variance_ratio_` turns per-component fractions into a running total.

Cumulative Curve

Plot cumulative variance vs component count; the "elbow" where the curve flattens near 0.95 suggests the optimal number of components to retain.

### Method 2: Scree Plot

This is like looking at the "steepness" of the information gain:

#### Scree plot (per-component variance)

PCA and Per-component Variance

Fit full PCA and read `explained_variance_ratio_`, each entry is the fraction of total variance captured by that one PC, unlike the cumulative curve which sums progressively.

Scree Plot

Plot individual variance per component; the "elbow" where the curve sharply levels off identifies the cutoff after which additional components add little information.

## Common Mistakes to Avoid

1. **Not Scaling Data**: Always standardize your data before PCA
2. **Using Too Many Components**: Don't keep components that don't add much information
3. **Ignoring the Context**: Make sure PCA makes sense for your specific problem

## Best Practices

1. **Always Scale Your Data**:

#### Preprocess helper: impute NaNs, then scale

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_for_pca(X):
    # Remove missing values
    X = np.nan_to_num(X)

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled
```

2. **Validate Your Results**:

#### Train/test reconstruction error

Fit on Train Only

Fit PCA on the training split only; apply `transform` (not `fit_transform`) to the test set so PCA learns the principal directions from training data alone, preventing data leakage.

Reconstruction MSE

Call `inverse_transform` on both sets and compute MSE; a much larger test error than train error suggests the chosen `n_components` over-fits the training distribution.

## When to Use PCA

1. **Data Visualization**: When you need to visualize high-dimensional data
2. **Feature Reduction**: When you have too many features
3. **Noise Reduction**: When your data has a lot of noise
4. **Data Compression**: When you need to reduce storage requirements

## Gotchas

* **PCA on unscaled data is dominated by high-variance features**: a feature measured in thousands will produce a principal component that is almost entirely that feature. Always apply `StandardScaler` before `PCA`, unless the features are already in the same units and you deliberately want to weight by raw variance.
* **`fit_transform` on the full dataset leaks test information**: calling `pca.fit_transform(X_all)` before splitting means the PCA directions are computed using test data. Fit PCA on training data only and use `pca.transform(X_test)` to avoid data leakage.
* **Explained variance ratio does not equal model performance**: keeping 95% of variance sounds safe, but the discarded 5% may contain exactly the signal a downstream classifier needs. Treat the variance threshold as a starting point and validate by measuring downstream task performance at several k values.
* **PC axes have no interpretable unit after projection**: the numbers in `X_pca` are coordinates in an abstract rotated space, not original feature values. Avoid statements like "PC1 is income", instead, inspect `pca.components_` loadings to understand which original features contribute most.
* **`inverse_transform` does not recover the original data exactly**: unless you keep all components, the reconstruction is lossy. The pixels look similar but are not identical; reconstruction error (MSE) quantifies how much information was dropped.
* **PCA is linear, it cannot capture non-linear structure**: if your data lies on a curved manifold (e.g., a Swiss roll), PCA will produce a poor low-dimensional embedding; use kernel PCA, t-SNE, or UMAP instead.

## Further Reading

1. [Scikit-learn PCA Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
2. [Understanding PCA with Python](https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60)
3. [Interactive PCA Visualization](https://setosa.io/ev/principal-component-analysis/)

## Practice Exercise

Try applying PCA to the famous Iris dataset:

1. Load the data
2. Standardize it
3. Apply PCA
4. Visualize the results
5. Compare the original and reduced features

Remember: The goal is to understand your data better, not just to reduce dimensions!
