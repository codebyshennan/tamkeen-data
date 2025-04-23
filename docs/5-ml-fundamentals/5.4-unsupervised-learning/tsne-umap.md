# t-SNE and UMAP: Visualizing Complex Data in 2D

Imagine you're trying to create a map of your neighborhood. You want to show how close different places are to each other, but you also want to preserve the relationships between them. That's exactly what t-SNE and UMAP do with high-dimensional data - they help us create meaningful 2D maps of complex data while preserving important relationships!

## What are t-SNE and UMAP? 🤔

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

Think of t-SNE as a smart photographer who knows exactly which angle to take a photo from to show the most important relationships between people in a group photo.

### UMAP (Uniform Manifold Approximation and Projection)

UMAP is like a more efficient version of t-SNE - it's like having a GPS that can create a simplified map of a complex city while still showing all the important connections between places.

## Why Do We Need These Tools? 💡

1. **Complex Data Visualization**: When we have data with many features, it's hard to see patterns. These tools help us visualize it in 2D.

2. **Preserving Local Structure**: They help us see how similar items are to each other, like showing which products are often bought together.

3. **Exploratory Analysis**: They're great for discovering patterns and relationships in your data.

## How Do They Work? 🛠️

Let's break it down with a simple example:

```python
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
```

## Real-World Example: Visualizing Handwritten Digits 📝

Let's see how these tools can help us visualize complex data:

```python
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
```

## Key Differences Between t-SNE and UMAP 🔍

1. **Speed**: UMAP is generally faster than t-SNE
2. **Memory Usage**: UMAP uses less memory
3. **Parameter Sensitivity**: t-SNE is more sensitive to parameter choices
4. **Global Structure**: UMAP often preserves global structure better

## When to Use Each Tool 🌟

### Use t-SNE when

- You need highly detailed local structure
- You have a small to medium dataset
- You want to focus on local relationships

### Use UMAP when

- You have a large dataset
- You need to preserve both local and global structure
- You need faster computation
- You want to use the embedding for downstream tasks

## Best Practices ✅

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

```python
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
```

## Common Mistakes to Avoid 🚫

1. **Not Scaling Data**: Always standardize your data first
2. **Using Wrong Parameters**: Choose parameters based on your data size
3. **Interpreting Distances**: Remember that distances in the visualization are not always meaningful
4. **Over-interpreting Results**: These are visualization tools, not clustering algorithms

## Further Reading 📚

1. [t-SNE Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
2. [UMAP Documentation](https://umap-learn.readthedocs.io/)
3. [Interactive t-SNE Visualization](https://distill.pub/2016/misread-tsne/)

## Practice Exercise 🎯

Try visualizing the famous MNIST dataset:

1. Load the data
2. Preprocess it
3. Apply both t-SNE and UMAP
4. Compare the results
5. Try different parameters to see how they affect the visualization

Remember: The goal is to understand your data better, not just to create pretty pictures!
