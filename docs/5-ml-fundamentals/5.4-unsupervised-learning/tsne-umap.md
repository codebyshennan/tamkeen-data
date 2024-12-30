# t-SNE and UMAP

While PCA is great for linear dimensionality reduction, real-world data often has complex, non-linear relationships. t-SNE and UMAP are powerful techniques that can capture these relationships and create meaningful visualizations. Let's explore these advanced methods! 🎨

## Understanding t-SNE 🎯

t-SNE (t-Distributed Stochastic Neighbor Embedding) works by:
1. Converting high-dimensional distances to probabilities
2. Creating a similar probability distribution in low dimensions
3. Minimizing the difference between these distributions

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# Load and prepare data
digits = load_digits()
X, y = digits.data, digits.target

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                     c=y, cmap='tab10')
plt.colorbar(scatter)
plt.title('t-SNE visualization of digits dataset')
plt.show()
```

## Understanding UMAP 🌍

UMAP (Uniform Manifold Approximation and Projection) is similar to t-SNE but:
- Is faster
- Better preserves global structure
- Has stronger theoretical foundations

```python
import umap

# Apply UMAP
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X)

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], 
                     c=y, cmap='tab10')
plt.colorbar(scatter)
plt.title('UMAP visualization of digits dataset')
plt.show()
```

## Comparing Methods 📊

Let's compare PCA, t-SNE, and UMAP on the same dataset:

```python
from sklearn.decomposition import PCA

def plot_dimensionality_reduction_comparison(X, y, figsize=(15, 5)):
    # Apply all three methods
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2)
    reducer = umap.UMAP()
    
    X_pca = pca.fit_transform(X)
    X_tsne = tsne.fit_transform(X)
    X_umap = reducer.fit_transform(X)
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # PCA
    scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=y, cmap='tab10')
    axes[0].set_title('PCA')
    
    # t-SNE
    axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                   c=y, cmap='tab10')
    axes[1].set_title('t-SNE')
    
    # UMAP
    axes[2].scatter(X_umap[:, 0], X_umap[:, 1], 
                   c=y, cmap='tab10')
    axes[2].set_title('UMAP')
    
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()

plot_dimensionality_reduction_comparison(X, y)
```

## Parameter Tuning 🎛️

### t-SNE Parameters
```python
def explore_tsne_parameters(X, y, perplexities=[5, 30, 50, 100]):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    for idx, perp in enumerate(perplexities):
        tsne = TSNE(n_components=2, perplexity=perp, 
                    random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                         c=y, cmap='tab10')
        axes[idx].set_title(f'Perplexity: {perp}')
    
    plt.tight_layout()
    plt.show()

explore_tsne_parameters(X, y)
```

### UMAP Parameters
```python
def explore_umap_parameters(X, y, n_neighbors=[5, 15, 30, 50],
                          min_dist=[0.1, 0.25, 0.5, 0.8]):
    fig, axes = plt.subplots(len(n_neighbors), len(min_dist), 
                            figsize=(20, 20))
    
    for i, nn in enumerate(n_neighbors):
        for j, md in enumerate(min_dist):
            reducer = umap.UMAP(n_neighbors=nn, min_dist=md,
                              random_state=42)
            X_umap = reducer.fit_transform(X)
            
            axes[i, j].scatter(X_umap[:, 0], X_umap[:, 1], 
                             c=y, cmap='tab10', s=5)
            axes[i, j].set_title(f'n_neighbors={nn}, min_dist={md}')
    
    plt.tight_layout()
    plt.show()

explore_umap_parameters(X, y)
```

## Real-World Applications 🌟

### 1. Single-Cell RNA Sequencing
```python
# Simulated gene expression data
n_cells = 1000
n_genes = 50
np.random.seed(42)

# Create three cell types
cell_type_1 = np.random.normal(0, 1, (n_cells//3, n_genes))
cell_type_2 = np.random.normal(3, 1, (n_cells//3, n_genes))
cell_type_3 = np.random.normal(-3, 1, (n_cells//3, n_genes))

X_genes = np.vstack([cell_type_1, cell_type_2, cell_type_3])
y_genes = np.repeat([0, 1, 2], n_cells//3)

# Apply UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
X_umap = reducer.fit_transform(X_genes)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], 
                     c=y_genes, cmap='tab10')
plt.colorbar(scatter)
plt.title('UMAP visualization of gene expression data')
plt.show()
```

### 2. Image Similarity
```python
from sklearn.datasets import load_digits
import numpy as np

# Load digits data
digits = load_digits()
X = digits.data
y = digits.target

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30)
X_tsne = tsne.fit_transform(X)

# Plot with thumbnail images
def plot_digits_tsne(X_tsne, images, y):
    fig, ax = plt.subplots(figsize=(15, 15))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10')
    
    # Add thumbnails for some points
    for idx in np.random.choice(len(images), 20):
        img = images[idx].reshape(8, 8)
        imagebox = OffsetImage(img, zoom=1, cmap='gray')
        ab = AnnotationBbox(imagebox, X_tsne[idx], frameon=False)
        ax.add_artist(ab)
    
    plt.colorbar(scatter)
    plt.title('t-SNE visualization with digit thumbnails')
    plt.show()

plot_digits_tsne(X_tsne, digits.images, y)
```

## Best Practices 🌟

### 1. Preprocessing
```python
from sklearn.preprocessing import StandardScaler

def preprocess_for_embedding(X):
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reduce noise with PCA if needed
    if X.shape[1] > 50:
        pca = PCA(n_components=50)
        X_scaled = pca.fit_transform(X_scaled)
    
    return X_scaled
```

### 2. Parameter Selection
```python
def find_best_params(X, y, method='tsne'):
    if method == 'tsne':
        perplexities = [5, 30, 50]
        scores = []
        
        for perp in perplexities:
            tsne = TSNE(perplexity=perp)
            X_embedded = tsne.fit_transform(X)
            # Calculate some metric (e.g., clustering score)
            score = silhouette_score(X_embedded, y)
            scores.append(score)
            
        best_perp = perplexities[np.argmax(scores)]
        return best_perp
    
    elif method == 'umap':
        # Similar process for UMAP parameters
        pass
```

### 3. Visualization
```python
def plot_embedding_with_confidence(X_embedded, y, 
                                 confidence_scores=None):
    plt.figure(figsize=(10, 8))
    
    if confidence_scores is None:
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                   c=y, cmap='tab10')
    else:
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
                            c=y, cmap='tab10', 
                            alpha=confidence_scores)
        plt.colorbar(scatter)
    
    plt.title('Embedding with confidence visualization')
    plt.show()
```

## Common Pitfalls and Solutions 🚧

1. **Computational Cost**
   - Use UMAP for large datasets
   - Apply PCA first to reduce dimensions
   - Use approximate nearest neighbors

2. **Reproducibility**
   - Set random state
   - Save transformed coordinates
   - Document parameters

3. **Interpretation**
   - Don't trust distances too much
   - Consider multiple runs
   - Compare with other methods

## Next Steps

Now that you understand dimensionality reduction, let's explore [Clustering Algorithms](./clustering.md) to find natural groups in your data!
