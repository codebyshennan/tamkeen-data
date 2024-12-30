# Principal Component Analysis (PCA)

Imagine trying to draw a 3D object on a 2D paper - you need to find the best angle to capture the most important features. That's exactly what PCA does with high-dimensional data! Let's learn how to reduce dimensions while preserving the most important information. 📊

## Understanding PCA 🎯

PCA works by:
1. Finding directions (principal components) of maximum variance
2. Projecting data onto these directions
3. Keeping only the most important components

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Create sample 2D data with correlation
np.random.seed(42)
n_samples = 300
t = np.random.uniform(0, 2*np.pi, n_samples)
x = np.cos(t) + np.random.normal(0, 0.1, n_samples)
y = np.sin(t) + np.random.normal(0, 0.1, n_samples)
data = np.column_stack((x, y))

# Standardize data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply PCA
pca = PCA()
data_pca = pca.fit_transform(data_scaled)

# Plot original data and principal components
plt.figure(figsize=(12, 5))

# Original data
plt.subplot(121)
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], alpha=0.5)
for i, (comp1, comp2) in enumerate(zip(pca.components_[0], pca.components_[1])):
    plt.arrow(0, 0, comp1, comp2, color='r', alpha=0.8, 
              head_width=0.05, head_length=0.1)
plt.title('Original Data with Principal Components')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Transformed data
plt.subplot(122)
plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5)
plt.title('Data in Principal Component Space')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

plt.tight_layout()
plt.show()

# Print explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)
```

## Real-World Example: Image Compression 🖼️

```python
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt

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
plt.show()
```

## Choosing Number of Components 📈

### 1. Explained Variance Ratio
```python
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
    plt.show()

plot_explained_variance(X)
```

### 2. Scree Plot
```python
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
    plt.show()

plot_scree(X)
```

## Feature Importance Analysis 🔍

```python
def plot_feature_importance(pca, feature_names):
    # Get absolute value of loadings
    loadings = np.abs(pca.components_)
    
    plt.figure(figsize=(12, 6))
    for i in range(2):  # Plot first two components
        plt.subplot(1, 2, i+1)
        plt.bar(range(len(feature_names)), loadings[i])
        plt.xticks(range(len(feature_names)), feature_names, rotation=45)
        plt.title(f'PC{i+1} Feature Importance')
    
    plt.tight_layout()
    plt.show()

# Example with iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
pca = PCA()
pca.fit(iris.data)
plot_feature_importance(pca, iris.feature_names)
```

## Practical Applications 🌟

### 1. Dimensionality Reduction for Visualization
```python
from sklearn.datasets import load_breast_cancer

# Load data
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
                     cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Breast Cancer Dataset in 2D')
plt.show()
```

### 2. Feature Engineering
```python
# Add PCA features to original dataset
def add_pca_features(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    
    return np.hstack([X, X_pca])
```

## Best Practices 🎯

### 1. Data Preprocessing
```python
# Always scale your data
def preprocess_for_pca(X):
    # Remove missing values
    X = np.nan_to_num(X)
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled
```

### 2. Handling Outliers
```python
from scipy import stats

def remove_outliers(X, z_threshold=3):
    z_scores = stats.zscore(X)
    mask = np.all(np.abs(z_scores) < z_threshold, axis=1)
    return X[mask]
```

### 3. Validation
```python
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
```

## Common Pitfalls and Solutions 🚧

1. **Not Scaling Data**
   - Always standardize features
   - Consider robust scaling for outliers
   - Check feature distributions

2. **Choosing Components**
   - Don't just use arbitrary numbers
   - Consider explained variance
   - Validate reconstruction error

3. **Interpretation**
   - Remember PCA is linear
   - Components may not be interpretable
   - Consider domain knowledge

## Next Steps

Now that you understand PCA, let's explore [t-SNE and UMAP](./tsne-umap.md) for non-linear dimensionality reduction!
