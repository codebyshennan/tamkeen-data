---
reading_minutes: 22
objectives:
  - "Define principal components as orthogonal directions of maximum variance and connect them to eigenvectors of the covariance matrix."
  - "Standardise before fitting `PCA`, then read `explained_variance_ratio_` and its cumulative sum to choose `n_components`."
  - "Use `fit_transform` and `inverse_transform` for projection, reconstruction, and image-style compression — and treat reconstruction MSE as the lossiness measure."
  - "Avoid leakage and over-interpretation: fit PCA on training data only, never name a PC after a single feature, and switch to non-linear methods (kernel PCA, t-SNE, UMAP) when the manifold is curved."
---

# Principal Component Analysis (PCA): Simplifying Complex Data

**After this lesson:** you can explain the core ideas in “Principal Component Analysis (PCA): Simplifying Complex Data” and reproduce the examples here in your own notebook or environment.

## Overview

**PCA** finds orthogonal **principal components**—directions of maximum variance—and lets you project data onto the top few for visualization, denoising, or as inputs to other models. **Prerequisites:** vectors and eigenvalue intuition from Module 1 linear algebra; [unsupervised learning hub](README.md).

Imagine you're trying to describe a person to someone who's never met them. Instead of listing every single detail (height, weight, hair color, eye color, clothing, etc.), you might focus on the most distinctive features that make them recognizable. That's exactly what PCA does with data - it helps us focus on the most important aspects while simplifying the rest!

## Helpful video

StatQuest: Principal Component Analysis (PCA), step by step.

<iframe width="560" height="315" src="https://www.youtube.com/embed/FgakZw6y1cY" title="StatQuest: Principal Component Analysis (PCA), Step-by-Step" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## What is PCA?

PCA is like creating a simplified map of a complex city. Just as a map helps you navigate a city by showing the most important streets and landmarks, PCA helps you navigate complex data by showing the most important features.

### Why Do We Need PCA?

1. **Too Many Features**: Imagine trying to understand a person by looking at 100 different measurements. It's overwhelming! PCA helps us focus on the most important ones.

2. **Visualization**: It's hard to visualize data with more than 3 dimensions. PCA helps us see patterns in high-dimensional data by reducing it to 2D or 3D.

3. **Noise Reduction**: Like removing background noise from a recording, PCA helps us focus on the important signals in our data.

## How Does PCA Work?

Let's break it down into simple steps:

1. **Standardize the Data**: First, we make sure all features are on the same scale (like converting different currencies to dollars).

2. **Find Principal Components**: These are like the main directions in which our data varies the most.

3. **Project the Data**: We rotate our data to align with these main directions.

{% include mermaid-diagram.html src="5-ml-fundamentals/5.4-unsupervised-learning/diagrams/pca-1.mmd" %}

*The output components are **uncorrelated** — PC1 captures the most variance, PC2 the next most, and so on. The scree plot shows where adding more components stops being useful.*

Let's see this in action with a simple example:

#### 2D toy cloud: scale, fit PCA, three subplots

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Create sample data that forms a cloud of points
np.random.seed(42)
n_samples = 300
t = np.random.uniform(0, 2*np.pi, n_samples)
x = np.cos(t) + np.random.normal(0, 0.1, n_samples)
y = np.sin(t) + np.random.normal(0, 0.1, n_samples)
data = np.column_stack((x, y))

# Step 1: Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Step 2: Apply PCA
pca = PCA()
data_pca = pca.fit_transform(data_scaled)

# Create visualization
plt.figure(figsize=(15, 5))

# Original data
plt.subplot(131)
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], alpha=0.5)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Data with principal components
plt.subplot(132)
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], alpha=0.5)
for i, (comp1, comp2) in enumerate(zip(pca.components_[0], pca.components_[1])):
    plt.arrow(0, 0, comp1, comp2, color='r', alpha=0.8, 
              head_width=0.05, head_length=0.1)
plt.title('Data with Principal Components')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Transformed data
plt.subplot(133)
plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5)
plt.title('Data in Principal Component Space')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

plt.tight_layout()
plt.savefig('assets/pca_basic_example.png')
plt.close()

# Print explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_)
{% endhighlight %}
```
Explained variance ratio: [0.50565666 0.49434334]
```


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="14-16" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Scale before PCA</span>
    </div>
    <div class="code-callout__body">
      <p>PCA finds the directions of maximum <em>variance</em>. A feature measured in thousands will dominate over one measured in units, so <code>StandardScaler</code> is applied first to give every feature equal footing.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="18-20" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Fit and transform</span>
    </div>
    <div class="code-callout__body">
      <p><code>PCA()</code> with no argument keeps all components. <code>fit_transform</code> finds the principal directions <em>and</em> projects the data onto them in one step — equivalent to calling <code>fit</code> then <code>transform</code>.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="33-40" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Principal component arrows</span>
    </div>
    <div class="code-callout__body">
      <p><code>pca.components_</code> holds the principal directions as unit vectors. Drawing them as arrows on the original data shows <em>which way the data varies most</em> — the longer the effective spread, the more variance that component captures.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="42-54" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">PC space + explained variance</span>
    </div>
    <div class="code-callout__body">
      <p>The third subplot shows data in <strong>PC coordinates</strong>: axes are now orthogonal directions of maximum variance. <code>explained_variance_ratio_</code> says what fraction of total variance each PC accounts for — two roughly-equal values here confirm neither direction dominates.</p>
    </div>
  </div>
</aside>
</div>

## Real-World Example: Image Compression

Let's see how PCA can help compress images while maintaining quality:

#### Digits reconstruction vs number of components

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load digits dataset
digits = load_digits()
X = digits.data

# Apply PCA with different numbers of components
n_components_list = [10, 20, 50, 64]
fig, axes = plt.subplots(2, len(n_components_list), figsize=(15, 6))

# Original image
sample_digit = X[0].reshape(8, 8)
for ax in axes[0]:
    ax.imshow(sample_digit, cmap='gray')
    ax.axis('off')
    ax.set_title('Original')

# Reconstructed images
for i, n_comp in enumerate(n_components_list):
    pca = PCA(n_components=n_comp)
    X_transformed = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)

    reconstructed_digit = X_reconstructed[0].reshape(8, 8)
    axes[1, i].imshow(reconstructed_digit, cmap='gray')
    axes[1, i].axis('off')
    axes[1, i].set_title(f'{n_comp} components\n{pca.explained_variance_ratio_.sum():.2%} var')

plt.tight_layout()
plt.savefig('assets/pca_image_compression.png')
plt.close()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-12" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Load Digits and Setup</span>
    </div>
    <div class="code-callout__body">
      <p>Load the 8×8 digit pixel dataset (64 features); a 2-row subplot grid will show the original image on top and reconstructions from increasing component counts below.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="14-18" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Original Row</span>
    </div>
    <div class="code-callout__body">
      <p>Fill the top row with the same original digit image four times as a reference; reshaping the flat 64-value array to (8, 8) is required for <code>imshow</code>.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="20-34" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Reconstruction Loop</span>
    </div>
    <div class="code-callout__body">
      <p>For each component count, fit PCA on all digits, project, then call <code>inverse_transform</code> to reconstruct; the title shows cumulative explained variance so readers see the quality-compression trade-off at a glance.</p>
    </div>
  </div>
</aside>
</div>

## How to Choose the Number of Components

### Method 1: Explained Variance Ratio

Think of this like a pie chart showing how much each component contributes to the total information:

#### Cumulative explained variance curve

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_explained_variance(X):
    pca = PCA()
    pca.fit(X)

    # Cumulative explained variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cumsum) + 1), cumsum, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance vs Number of Components')
    plt.grid(True)
    plt.savefig('assets/pca_explained_variance.png')
    plt.close()

plot_explained_variance(X)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-10" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Full PCA Fit</span>
    </div>
    <div class="code-callout__body">
      <p>Fit PCA with no component limit to get all eigenvalues; <code>np.cumsum</code> on <code>explained_variance_ratio_</code> turns per-component fractions into a running total.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="12-20" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Cumulative Curve</span>
    </div>
    <div class="code-callout__body">
      <p>Plot cumulative variance vs component count; the "elbow" where the curve flattens near 0.95 suggests the optimal number of components to retain.</p>
    </div>
  </div>
</aside>
</div>

### Method 2: Scree Plot

This is like looking at the "steepness" of the information gain:

#### Scree plot (per-component variance)

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_scree(X):
    pca = PCA()
    pca.fit(X)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             pca.explained_variance_ratio_, 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')
    plt.grid(True)
    plt.savefig('assets/pca_scree_plot.png')
    plt.close()

plot_scree(X)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">PCA and Per-component Variance</span>
    </div>
    <div class="code-callout__body">
      <p>Fit full PCA and read <code>explained_variance_ratio_</code> — each entry is the fraction of total variance captured by that one PC, unlike the cumulative curve which sums progressively.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="10-18" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Scree Plot</span>
    </div>
    <div class="code-callout__body">
      <p>Plot individual variance per component; the "elbow" where the curve sharply levels off identifies the cutoff after which additional components add little information.</p>
    </div>
  </div>
</aside>
</div>

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

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def validate_pca_results(X, n_components):
    # Split data
    X_train, X_test = train_test_split(X, test_size=0.2)

    # Fit PCA on training data
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Reconstruct data
    X_train_reconstructed = pca.inverse_transform(X_train_pca)
    X_test_reconstructed = pca.inverse_transform(X_test_pca)

    # Calculate reconstruction error
    train_error = np.mean((X_train - X_train_reconstructed) ** 2)
    test_error = np.mean((X_test - X_test_reconstructed) ** 2)

    return train_error, test_error
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-12" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Fit on Train Only</span>
    </div>
    <div class="code-callout__body">
      <p>Fit PCA on the training split only; apply <code>transform</code> (not <code>fit_transform</code>) to the test set so PCA learns the principal directions from training data alone — preventing data leakage.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="14-22" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Reconstruction MSE</span>
    </div>
    <div class="code-callout__body">
      <p>Call <code>inverse_transform</code> on both sets and compute MSE; a much larger test error than train error suggests the chosen <code>n_components</code> over-fits the training distribution.</p>
    </div>
  </div>
</aside>
</div>

## When to Use PCA

1. **Data Visualization**: When you need to visualize high-dimensional data
2. **Feature Reduction**: When you have too many features
3. **Noise Reduction**: When your data has a lot of noise
4. **Data Compression**: When you need to reduce storage requirements

## Gotchas

- **PCA on unscaled data is dominated by high-variance features** — a feature measured in thousands will produce a principal component that is almost entirely that feature. Always apply `StandardScaler` before `PCA`, unless the features are already in the same units and you deliberately want to weight by raw variance.
- **`fit_transform` on the full dataset leaks test information** — calling `pca.fit_transform(X_all)` before splitting means the PCA directions are computed using test data. Fit PCA on training data only and use `pca.transform(X_test)` to avoid data leakage.
- **Explained variance ratio does not equal model performance** — keeping 95% of variance sounds safe, but the discarded 5% may contain exactly the signal a downstream classifier needs. Treat the variance threshold as a starting point and validate by measuring downstream task performance at several k values.
- **PC axes have no interpretable unit after projection** — the numbers in `X_pca` are coordinates in an abstract rotated space, not original feature values. Avoid statements like "PC1 is income" — instead, inspect `pca.components_` loadings to understand which original features contribute most.
- **`inverse_transform` does not recover the original data exactly** — unless you keep all components, the reconstruction is lossy. The pixels look similar but are not identical; reconstruction error (MSE) quantifies how much information was dropped.
- **PCA is linear — it cannot capture non-linear structure** — if your data lies on a curved manifold (e.g., a Swiss roll), PCA will produce a poor low-dimensional embedding; use kernel PCA, t-SNE, or UMAP instead.

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
