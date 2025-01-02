# Advanced SVM Techniques 🚀

Let's explore advanced concepts and optimizations that can improve your SVM implementations.

## Advanced Optimization Techniques 🔧

### Sequential Minimal Optimization (SMO)

> **SMO** is an efficient algorithm for solving the SVM optimization problem by breaking it down into smaller, manageable pieces.

```python
from sklearn.svm import SVC
import numpy as np

class OptimizedSVM:
    def __init__(self, max_iter=1000):
        self.max_iter = max_iter
        self.model = SVC(
            kernel='rbf',
            cache_size=1000,  # Increase cache for faster training
            max_iter=max_iter
        )
        
    def fit_with_early_stopping(self, X, y, tolerance=1e-3):
        """Train with early stopping based on convergence"""
        prev_score = float('-inf')
        for i in range(self.max_iter):
            self.model.max_iter = i + 1
            self.model.fit(X, y)
            score = self.model.score(X, y)
            
            if abs(score - prev_score) < tolerance:
                print(f"Converged after {i+1} iterations")
                break
                
            prev_score = score
```

## Advanced Kernel Techniques 🎯

### Custom Kernel Implementation

```python
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

class CustomKernelSVM:
    def __init__(self):
        self.model = SVC(kernel='precomputed')
        
    def custom_kernel(self, X, Y=None):
        """Implement custom kernel function"""
        if Y is None:
            Y = X
            
        # Example: Combination of RBF and Polynomial
        gamma = 0.1
        degree = 2
        
        # RBF component
        rbf = np.exp(-gamma * 
                     pairwise_kernels(X, Y, metric='euclidean')**2)
        
        # Polynomial component
        poly = (np.dot(X, Y.T) + 1) ** degree
        
        return 0.7 * rbf + 0.3 * poly
        
    def fit(self, X, y):
        """Train model with custom kernel"""
        K = self.custom_kernel(X)
        self.model.fit(K, y)
        self.X_train = X
        
    def predict(self, X):
        """Make predictions using custom kernel"""
        K = self.custom_kernel(X, self.X_train)
        return self.model.predict(K)
```

## Advanced Visualization 📊

### Decision Boundary Visualization

```python
import matplotlib.pyplot as plt

class SVMVisualizer:
    def __init__(self, model):
        self.model = model
        
    def plot_decision_boundary(self, X, y):
        """Visualize decision boundary and margins"""
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.02),
            np.arange(y_min, y_max, 0.02)
        )
        
        # Get predictions
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.figure(figsize=(12, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        
        # Highlight support vectors
        plt.scatter(
            self.model.support_vectors_[:, 0],
            self.model.support_vectors_[:, 1],
            s=200, linewidth=1, facecolors='none',
            edgecolors='k', label='Support Vectors'
        )
        
        plt.title('SVM Decision Boundary')
        plt.legend()
        plt.show()
        
    def plot_margin_width(self, X, y):
        """Visualize margin width"""
        # Get support vectors
        sv = self.model.support_vectors_
        
        # Calculate margin width
        w = self.model.coef_[0]
        margin_width = 2 / np.sqrt(np.sum(w ** 2))
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6)
        
        # Plot margin lines
        a = -w[0] / w[1]
        xx = np.linspace(X[:, 0].min(), X[:, 0].max())
        yy = a * xx - (self.model.intercept_[0]) / w[1]
        
        plt.plot(xx, yy, 'k-', label='Decision Boundary')
        plt.plot(xx, yy + margin_width, 'k--', label='Margin')
        plt.plot(xx, yy - margin_width, 'k--')
        
        plt.title(f'SVM Margin Width: {margin_width:.2f}')
        plt.legend()
        plt.show()
```

## Performance Optimization 🏃‍♂️

### Memory-Efficient Implementation

```python
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import numpy as np

class MemoryEfficientSVM:
    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size
        self.scaler = StandardScaler()
        self.model = LinearSVC(
            dual=False,  # More memory efficient
            max_iter=1000
        )
        
    def fit(self, X, y):
        """Train model in chunks"""
        # Scale features using partial_fit
        for i in range(0, len(X), self.chunk_size):
            chunk = X[i:i + self.chunk_size]
            self.scaler.partial_fit(chunk)
            
        # Train model on scaled data
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
```

### Parallel Processing

```python
from joblib import Parallel, delayed

class ParallelSVM:
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs
        
    def parallel_grid_search(self, X, y):
        """Parallel parameter search"""
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': ['rbf', 'linear']
        }
        
        def evaluate_params(params):
            model = SVC(**params)
            scores = cross_val_score(
                model, X, y,
                cv=5, n_jobs=1
            )
            return params, scores.mean()
            
        # Generate parameter combinations
        param_combinations = [
            dict(zip(param_grid.keys(), v)) 
            for v in product(*param_grid.values())
        ]
        
        # Parallel evaluation
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_params)(params)
            for params in param_combinations
        )
        
        # Find best parameters
        best_params = max(results, key=lambda x: x[1])[0]
        return best_params
```

## Advanced Feature Engineering 🛠️

### Feature Selection with SVM

```python
from sklearn.feature_selection import SelectFromModel

class SVMFeatureSelector:
    def __init__(self, threshold='mean'):
        self.threshold = threshold
        self.model = LinearSVC(
            penalty='l1',
            dual=False,
            max_iter=1000
        )
        
    def select_features(self, X, y):
        """Select important features using SVM weights"""
        # Train linear SVM
        self.model.fit(X, y)
        
        # Create feature selector
        selector = SelectFromModel(
            self.model,
            prefit=True,
            threshold=self.threshold
        )
        
        # Get selected feature mask
        feature_mask = selector.get_support()
        
        # Transform data
        X_selected = selector.transform(X)
        
        return X_selected, feature_mask
        
    def analyze_importance(self, feature_names):
        """Analyze feature importance"""
        importances = np.abs(self.model.coef_[0])
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(len(importances)),
                importances[indices])
        plt.xticks(range(len(importances)),
                   [feature_names[i] for i in indices],
                   rotation=45)
        plt.tight_layout()
        plt.show()
```

## Online Learning 🔄

### Incremental SVM

```python
from sklearn.linear_model import SGDClassifier

class IncrementalSVM:
    def __init__(self):
        self.model = SGDClassifier(
            loss='hinge',  # SVM loss
            learning_rate='optimal',
            max_iter=1000
        )
        
    def partial_fit(self, X, y, classes=None):
        """Incrementally train the model"""
        self.model.partial_fit(X, y, classes=classes)
        
    def evaluate_stream(self, X_stream, y_stream, batch_size=100):
        """Evaluate on streaming data"""
        accuracies = []
        
        for i in range(0, len(X_stream), batch_size):
            X_batch = X_stream[i:i + batch_size]
            y_batch = y_stream[i:i + batch_size]
            
            # Predict before training
            accuracy = self.model.score(X_batch, y_batch)
            accuracies.append(accuracy)
            
            # Update model
            self.partial_fit(X_batch, y_batch)
            
        return accuracies
```

## Next Steps 📚

After mastering these advanced techniques:
1. Learn about [real-world applications](5-applications.md)
2. Practice implementing these optimizations
3. Experiment with custom kernels
4. Benchmark different approaches

Remember:
- Profile code before optimizing
- Test thoroughly after modifications
- Document performance improvements
- Consider trade-offs between speed and accuracy
