---
reading_minutes: 18
objectives:
  - "Compare t-SNE and UMAP on what they preserve (local neighbourhoods) and how they differ on speed, global structure, and `transform` support for new points."
  - "Preprocess high-dimensional data before either method: scale, then optionally PCA down to ~50 dimensions to cut runtime and noise."
  - "Run both on the same data and tune key parameters — `perplexity` for t-SNE, `n_neighbors` / `min_dist` for UMAP — by checking visual stability across seeds."
  - "Avoid over-interpreting embeddings: cluster sizes and inter-cluster distances are not meaningful, and UMAP can find structure even in pure noise."
---

# t-SNE and UMAP: Visualizing Complex Data in 2D

**After this lesson:** you can explain the core ideas in “t-SNE and UMAP: Visualizing Complex Data in 2D” and reproduce the examples here in your own notebook or environment.

## Overview

**t-SNE** and **UMAP** for visualization: perplexity, local vs global structure, and when a 2D plot misleads.

[PCA](pca.md) for linear structure first; [unsupervised README](README.md) for ordering.


## What are t-SNE and UMAP?

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

Think of t-SNE as a smart photographer who knows exactly which angle to take a photo from to show the most important relationships between people in a group photo.

### UMAP (Uniform Manifold Approximation and Projection)

UMAP is like a more efficient version of t-SNE - it's like having a GPS that can create a simplified map of a complex city while still showing all the important connections between places.

## Why Do We Need These Tools?

1. **Complex Data Visualization**: When we have data with many features, it's hard to see patterns. These tools help us visualize it in 2D.

2. **Preserving Local Structure**: They help us see how similar items are to each other, like showing which products are often bought together.

3. **Exploratory Analysis**: They're great for discovering patterns and relationships in your data.

## How Do They Work?

{% include mermaid-diagram.html src="5-ml-fundamentals/5.4-unsupervised-learning/diagrams/tsne-umap-1.mmd" %}

*Always run PCA first to reduce to ~50 dimensions before feeding into t-SNE — it speeds up computation significantly and removes noise.*

Let's break it down with a simple example:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
import umap

# Create sample data with clear clusters
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Apply UMAP
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X)

# Create visualization
plt.figure(figsize=(15, 5))

# Original data (first two dimensions)
plt.subplot(131)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Original Data (First 2 Dimensions)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# t-SNE visualization
plt.subplot(132)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

# UMAP visualization
plt.subplot(133)
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis')
plt.title('UMAP Visualization')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')

plt.tight_layout()
plt.savefig('assets/tsne_umap_comparison.png')
plt.close()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-16" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Data, t-SNE, and UMAP Fits</span>
    </div>
    <div class="code-callout__body">
      <p>Generate four well-separated blobs then project with both t-SNE and UMAP; <code>random_state=42</code> makes results reproducible — t-SNE especially varies across runs without fixing the seed.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="18-44" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Three-panel Comparison</span>
    </div>
    <div class="code-callout__body">
      <p>Left shows original 2D space; middle is t-SNE; right is UMAP — comparing all three side by side lets you see how each method transforms the cluster structure.</p>
    </div>
  </div>
</aside>
</div>

## Real-World Example: Visualizing Handwritten Digits

Let's see how these tools can help us visualize complex data:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt

# Load digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Apply UMAP
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X)

# Create visualization
plt.figure(figsize=(15, 5))

# t-SNE visualization
plt.subplot(121)
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10')
plt.colorbar(scatter)
plt.title('t-SNE Visualization of Digits')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

# UMAP visualization
plt.subplot(122)
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10')
plt.colorbar(scatter)
plt.title('UMAP Visualization of Digits')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')

plt.tight_layout()
plt.savefig('assets/tsne_umap_digits.png')
plt.close()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-15" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Load Digits and Embed</span>
    </div>
    <div class="code-callout__body">
      <p>Load the 64-feature digit pixel dataset and project to 2D with both t-SNE and UMAP; the 10-class target <code>y</code> provides color labels to assess how well each method separates digit classes.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="17-38" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Side-by-side Digit Maps</span>
    </div>
    <div class="code-callout__body">
      <p>Use <code>tab10</code> colormap for 10 distinct digit classes; the colorbar maps colors to digit values — clean separation of color clusters indicates the embedding preserved class structure.</p>
    </div>
  </div>
</aside>
</div>

## Key Differences Between t-SNE and UMAP

1. **Speed**: UMAP is generally faster than t-SNE
2. **Memory Usage**: UMAP uses less memory
3. **Parameter Sensitivity**: t-SNE is more sensitive to parameter choices
4. **Global Structure**: UMAP often preserves global structure better

## When to Use Each Tool

### Use t-SNE when

- You need highly detailed local structure
- You have a small to medium dataset
- You want to focus on local relationships

### Use UMAP when

- You have a large dataset
- You need to preserve both local and global structure
- You need faster computation
- You want to use the embedding for downstream tasks

## Best Practices

1. **Preprocessing**:

```python
def preprocess_for_visualization(X):
    # Remove missing values
    X = np.nan_to_num(X)
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled
```

2. **Parameter Tuning**:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def find_best_parameters(X, y):
    # Try different perplexity values for t-SNE
    perplexities = [5, 30, 50, 100]
    plt.figure(figsize=(15, 10))

    for i, perplexity in enumerate(perplexities):
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X)

        plt.subplot(2, 2, i+1)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
        plt.title(f't-SNE with perplexity={perplexity}')

    plt.tight_layout()
    plt.savefig('assets/tsne_parameter_tuning.png')
    plt.close()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-4" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Perplexity Grid</span>
    </div>
    <div class="code-callout__body">
      <p>Test four perplexity values from 5 to 100; low perplexity focuses on very local neighbors (fragmented clusters) while high perplexity captures more global structure.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="6-17" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Four-subplot Comparison</span>
    </div>
    <div class="code-callout__body">
      <p>Each subplot runs a fresh t-SNE with a different perplexity; the 2×2 grid lets you visually pick the perplexity that produces the clearest and most stable cluster structure for your data.</p>
    </div>
  </div>
</aside>
</div>

## Common Mistakes to Avoid

1. **Not Scaling Data**: Always standardize your data first
2. **Using Wrong Parameters**: Choose parameters based on your data size
3. **Interpreting Distances**: Remember that distances in the visualization are not always meaningful
4. **Over-interpreting Results**: These are visualization tools, not clustering algorithms

## Gotchas

- **Cluster sizes and inter-cluster distances in t-SNE plots are not meaningful** — t-SNE distorts global structure to preserve local neighborhoods; two clusters appearing close together or one cluster appearing larger than another does not mean they are more similar or more spread out in the original space.
- **Different runs of t-SNE produce different layouts** — even with `random_state` set, changing `perplexity` or the number of iterations yields a completely different-looking plot. Always fix the seed and treat the layout as one possible visualization, not a unique ground truth.
- **t-SNE does not support `transform` on new data** — unlike PCA or UMAP, sklearn's `TSNE` has no `transform` method; you must refit the entire embedding if a new point arrives, making it unsuitable for production pipelines.
- **Running t-SNE on raw high-dimensional data is slow and noisy** — the recommendation (noted in the lesson) to run PCA first down to ~50 dimensions is critical: skipping it on data with hundreds of features multiplies runtime significantly and can degrade embedding quality.
- **UMAP embeddings can look deceptively clean with random data** — UMAP is prone to finding structure even in pure noise; always verify that clusters visible in a UMAP plot correspond to real structure by checking cluster quality in the original feature space.
- **Perplexity must be smaller than the number of samples** — setting `perplexity` larger than `n_samples - 1` raises a `ValueError`; a rule of thumb is perplexity between 5 and 50 for most datasets.

## Further Reading

1. [t-SNE Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
2. [UMAP Documentation](https://umap-learn.readthedocs.io/)
3. [Interactive t-SNE Visualization](https://distill.pub/2016/misread-tsne/)

## Practice Exercise

Try visualizing the famous MNIST dataset:

1. Load the data
2. Preprocess it
3. Apply both t-SNE and UMAP
4. Compare the results
5. Try different parameters to see how they affect the visualization

Remember: The goal is to understand your data better, not just to create pretty pictures!
