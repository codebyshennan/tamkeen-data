# Advanced KNN Techniques 🚀

Let's explore advanced concepts and optimizations that can improve your KNN implementations.

## Weighted KNN 🎯

> **Weighted KNN** assigns different weights to neighbors based on their distance, giving closer neighbors more influence on the prediction.

### Distance-Based Weights

```python
import numpy as np
from scipy.spatial.distance import euclidean

class WeightedKNN:
    def __init__(self, k=3, weight_type='inverse'):
        self.k = k
        self.weight_type = weight_type
        
    def _calculate_weights(self, distances):
        """Calculate weights based on distances"""
        if self.weight_type == 'inverse':
            # Add small constant to avoid division by zero
            return 1 / (distances + 1e-8)
        elif self.weight_type == 'exponential':
            return np.exp(-distances)
        else:
            return np.ones_like(distances)
            
    def predict(self, X):
        distances = np.array([
            [euclidean(x, x_train) for x_train in self.X_train]
            for x in X
        ])
        
        # Get k nearest indices and their distances
        k_indices = np.argsort(distances, axis=1)[:, :self.k]
        k_distances = np.take_along_axis(distances, k_indices, axis=1)
        
        # Calculate weights
        weights = self._calculate_weights(k_distances)
        
        # Weighted prediction
        predictions = []
        for idx, weight in zip(k_indices, weights):
            k_labels = self.y_train[idx]
            weighted_votes = {
                label: np.sum(weight[k_labels == label])
                for label in np.unique(self.y_train)
            }
            predictions.append(max(weighted_votes.items(), 
                                key=lambda x: x[1])[0])
            
        return np.array(predictions)
```

## Dimensionality Reduction 📊

### PCA with Visualization

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class DimensionalityReducer:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        
    def fit_transform_plot(self, X, y):
        """Reduce dimensions and plot results"""
        # Reduce dimensions
        X_reduced = self.pca.fit_transform(X)
        
        # Plot results
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                            c=y, cmap='viridis')
        plt.colorbar(scatter)
        plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.2%})')
        plt.title('PCA Visualization of Dataset')
        plt.show()
        
        return X_reduced

# Example usage
reducer = DimensionalityReducer()
X_reduced = reducer.fit_transform_plot(X, y)
```

## Adaptive KNN 🔄

> **Adaptive KNN** adjusts the number of neighbors based on local data density.

```python
class AdaptiveKNN:
    def __init__(self, k_min=1, k_max=20):
        self.k_min = k_min
        self.k_max = k_max
        
    def _find_optimal_k(self, x, distances):
        """Find optimal k for a specific point"""
        # Calculate local density
        density = np.mean(distances[:self.k_max])
        
        # Adjust k based on density
        k = int(self.k_min + 
               (self.k_max - self.k_min) * 
               np.exp(-density))
        
        return max(self.k_min, min(k, self.k_max))
    
    def predict(self, X):
        predictions = []
        for x in X:
            # Calculate distances
            distances = np.array([
                euclidean(x, x_train) 
                for x_train in self.X_train
            ])
            
            # Find optimal k
            k = self._find_optimal_k(x, distances)
            
            # Get k nearest neighbors
            k_indices = np.argsort(distances)[:k]
            k_labels = self.y_train[k_indices]
            
            # Make prediction
            prediction = np.bincount(k_labels).argmax()
            predictions.append(prediction)
            
        return np.array(predictions)
```

## Advanced Tree Structures 🌳

### Ball Tree Implementation

```python
from sklearn.neighbors import BallTree

class OptimizedKNN:
    def __init__(self, k=5, leaf_size=30):
        self.k = k
        self.leaf_size = leaf_size
        
    def fit(self, X, y):
        """Build ball tree and store training data"""
        self.tree = BallTree(X, leaf_size=self.leaf_size)
        self.y_train = y
        
    def predict(self, X):
        """Make predictions using ball tree"""
        # Find k nearest neighbors
        distances, indices = self.tree.query(
            X, k=self.k, return_distance=True
        )
        
        # Get predictions
        predictions = []
        for idx_set in indices:
            k_labels = self.y_train[idx_set]
            prediction = np.bincount(k_labels).argmax()
            predictions.append(prediction)
            
        return np.array(predictions)
```

## Cross-Validation Visualization 📈

```python
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def visualize_k_selection(X, y, k_range=range(1, 31, 2)):
    """Visualize cross-validation scores for different k values"""
    # Calculate scores for different k values
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
        scores.append(score.mean())
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, scores, 'o-')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Cross-validation Accuracy')
    plt.title('Accuracy vs. k Value')
    
    # Add error bars
    plt.fill_between(k_range, 
                     [s - scores_std for s in scores],
                     [s + scores_std for s in scores],
                     alpha=0.2)
    
    plt.grid(True)
    plt.show()
    
    # Return best k
    best_k = k_range[np.argmax(scores)]
    print(f"Best k value: {best_k}")
    return best_k
```

## Feature Importance Analysis 🔍

```python
def analyze_feature_importance(X, y, feature_names):
    """Analyze importance of each feature"""
    importances = []
    base_score = cross_val_score(
        KNeighborsClassifier(), X, y, cv=5
    ).mean()
    
    for i in range(X.shape[1]):
        # Create copy without current feature
        X_reduced = np.delete(X, i, axis=1)
        
        # Calculate score without this feature
        score = cross_val_score(
            KNeighborsClassifier(), X_reduced, y, cv=5
        ).mean()
        
        # Importance is reduction in score
        importance = base_score - score
        importances.append(importance)
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, importances)
    plt.xticks(rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance Analysis')
    plt.tight_layout()
    plt.show()
    
    return dict(zip(feature_names, importances))
```

## Handling Imbalanced Data 📊

```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def create_balanced_pipeline():
    """Create pipeline with SMOTE and undersampling"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('oversample', SMOTE(sampling_strategy=0.5)),
        ('undersample', RandomUnderSampler(sampling_strategy=0.8)),
        ('knn', KNeighborsClassifier())
    ])
```

## Performance Optimization 🚀

### Parallel Processing

```python
from joblib import Parallel, delayed

class ParallelKNN:
    def __init__(self, k=5, n_jobs=-1):
        self.k = k
        self.n_jobs = n_jobs
        
    def _predict_batch(self, batch):
        """Predict batch of samples in parallel"""
        return np.array([self._predict_single(x) for x in batch])
        
    def predict(self, X):
        """Make predictions using parallel processing"""
        # Split data into batches
        n_samples = len(X)
        batch_size = max(1, n_samples // (4 * self.n_jobs))
        batches = [
            X[i:i + batch_size] 
            for i in range(0, n_samples, batch_size)
        ]
        
        # Parallel prediction
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(self._predict_batch)(batch) 
            for batch in batches
        )
        
        return np.concatenate(predictions)
```

## Next Steps 📚

Now that you understand advanced KNN techniques:
1. Learn about [real-world applications](5-applications.md)
2. Practice implementing these optimizations
3. Experiment with different combinations of techniques
4. Benchmark performance improvements

Remember:
- Not all optimizations are needed for every problem
- Start with simpler solutions first
- Profile your code to identify bottlenecks
- Test thoroughly after implementing optimizations
