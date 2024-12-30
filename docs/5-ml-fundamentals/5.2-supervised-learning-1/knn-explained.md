# k-Nearest Neighbors (kNN): A Visual Guide 🎯

## 1. Understanding Distance Metrics Visually

### Different Ways to Measure Distance
```mermaid
graph TD
    A[Distance Metrics] --> B[Euclidean]
    A --> C[Manhattan]
    A --> D[Minkowski]
    
    B --> E[2D Example:<br/>d = √((x₂-x₁)² + (y₂-y₁)²)]
    C --> F[2D Example:<br/>d = |x₂-x₁| + |y₂-y₁|]
    D --> G[2D Example:<br/>d = (|x₂-x₁|ᵖ + |y₂-y₁|ᵖ)^(1/p)]
```

### Visual Comparison of Distance Metrics
```python
import numpy as np
import matplotlib.pyplot as plt

def plot_distance_contours():
    """Visualize how different distance metrics create different shapes"""
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    
    # Calculate distances
    euclidean = np.sqrt(X**2 + Y**2)
    manhattan = np.abs(X) + np.abs(Y)
    minkowski = (np.abs(X)**3 + np.abs(Y)**3)**(1/3)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot contours
    ax1.contour(X, Y, euclidean)
    ax1.set_title('Euclidean Distance')
    
    ax2.contour(X, Y, manhattan)
    ax2.set_title('Manhattan Distance')
    
    ax3.contour(X, Y, minkowski)
    ax3.set_title('Minkowski Distance (p=3)')
    
    plt.tight_layout()
```

## 2. kNN Algorithm Step by Step

### Classification Process Visualization
```mermaid
graph LR
    A[New Point] --> B[Calculate Distances]
    B --> C[Find k Nearest<br/>Neighbors]
    C --> D[Vote for Class]
    D --> E[Assign Label]
    
    style A fill:#f9f,stroke:#333
    style B fill:#bbf,stroke:#333
    style C fill:#bbf,stroke:#333
    style D fill:#bbf,stroke:#333
    style E fill:#f9f,stroke:#333
```

### Interactive kNN Classifier
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

class InteractiveKNN:
    def __init__(self, k=3):
        self.k = k
        self.X = None
        self.y = None
        self.classifier = KNeighborsClassifier(n_neighbors=k)
        
    def plot_decision_boundary(self):
        """Plot the decision boundary and training points"""
        h = 0.02  # Step size
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Plot decision boundary
        Z = self.classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.4)
        
        # Plot training points
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, alpha=0.8)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'kNN Decision Boundary (k={self.k})')
        
    def highlight_neighbors(self, query_point):
        """Highlight k nearest neighbors for a query point"""
        distances, indices = self.classifier.kneighbors([query_point])
        
        plt.scatter(query_point[0], query_point[1], 
                   color='red', marker='*', s=200,
                   label='Query Point')
        
        # Plot circles around neighbors
        for neighbor_idx in indices[0]:
            plt.scatter(self.X[neighbor_idx, 0], 
                       self.X[neighbor_idx, 1],
                       s=300, facecolors='none', 
                       edgecolors='r', linewidth=2)
```

## 3. Effect of k on Decision Boundary

### Visual Comparison of Different k Values
```python
def compare_k_values(X, y, k_values=[1, 3, 7, 15]):
    """Compare decision boundaries for different k values"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    for idx, k in enumerate(k_values):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X, y)
        
        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        axes[idx].contourf(xx, yy, Z, alpha=0.4)
        axes[idx].scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        axes[idx].set_title(f'k = {k}')
```

## 4. Handling the Curse of Dimensionality

### Dimension Reduction Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class DimensionalityHandler:
    def __init__(self, n_components=0.95):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components)),
            ('knn', KNeighborsClassifier())
        ])
        
    def plot_explained_variance(self, X):
        """Plot cumulative explained variance ratio"""
        pca = PCA()
        pca.fit(X)
        
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Analysis')
        plt.grid(True)
        
    def visualize_reduced_data(self, X, y):
        """Visualize data in 2D after PCA"""
        X_reduced = self.pipeline.fit_transform(X)[:, :2]
        
        plt.figure(figsize=(8, 6))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Data After Dimensionality Reduction')
```

## 5. Optimization Techniques

### KD-Tree Implementation
```python
from sklearn.neighbors import KDTree
import numpy as np

class OptimizedKNN:
    def __init__(self, k=5):
        self.k = k
        self.tree = None
        self.y_train = None
        
    def build_tree(self, X):
        """Build KD-tree for efficient neighbor search"""
        self.tree = KDTree(X)
        
    def visualize_tree_structure(self, depth=3):
        """Visualize KD-tree partitioning (2D only)"""
        def plot_partition(bounds, depth=0, max_depth=3):
            if depth >= max_depth:
                return
                
            x_mid = (bounds[0] + bounds[2]) / 2
            y_mid = (bounds[1] + bounds[3]) / 2
            
            if depth % 2 == 0:
                plt.vlines(x_mid, bounds[1], bounds[3], 'r', alpha=0.5)
                plot_partition([bounds[0], bounds[1], x_mid, bounds[3]], 
                             depth + 1, max_depth)
                plot_partition([x_mid, bounds[1], bounds[2], bounds[3]], 
                             depth + 1, max_depth)
            else:
                plt.hlines(y_mid, bounds[0], bounds[2], 'b', alpha=0.5)
                plot_partition([bounds[0], bounds[1], bounds[2], y_mid], 
                             depth + 1, max_depth)
                plot_partition([bounds[0], y_mid, bounds[2], bounds[3]], 
                             depth + 1, max_depth)
        
        x_min, x_max = self.tree.data[:, 0].min(), self.tree.data[:, 0].max()
        y_min, y_max = self.tree.data[:, 1].min(), self.tree.data[:, 1].max()
        
        plt.figure(figsize=(10, 10))
        plt.scatter(self.tree.data[:, 0], self.tree.data[:, 1])
        plot_partition([x_min, y_min, x_max, y_max], max_depth=depth)
        plt.title('KD-tree Partitioning')
```

## 6. Real-world Applications

### Image Similarity Search
```python
from sklearn.neighbors import NearestNeighbors
from PIL import Image

class ImageSimilarityFinder:
    def __init__(self, k=5):
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=k)
        
    def preprocess_image(self, image):
        """Convert image to feature vector"""
        # Resize for consistency
        image = image.resize((64, 64))
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        # Flatten to 1D array
        return np.array(image).flatten()
        
    def fit(self, images):
        """Build index of images"""
        features = np.array([self.preprocess_image(img) for img in images])
        self.nn.fit(features)
        
    def find_similar(self, query_image):
        """Find k most similar images"""
        query_features = self.preprocess_image(query_image)
        distances, indices = self.nn.kneighbors([query_features])
        return indices[0], distances[0]
```

### Recommendation System
```python
class CollaborativeFilter:
    def __init__(self, k=5):
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=k, metric='cosine')
        
    def fit(self, user_item_matrix):
        """Build user similarity index"""
        self.user_item_matrix = user_item_matrix
        self.nn.fit(user_item_matrix)
        
    def get_recommendations(self, user_id):
        """Get recommendations for a user"""
        # Find similar users
        distances, indices = self.nn.kneighbors(
            [self.user_item_matrix[user_id]]
        )
        
        # Weight recommendations by similarity
        weights = 1 - distances.flatten()
        weighted_ratings = np.zeros_like(self.user_item_matrix[0])
        
        for idx, weight in zip(indices.flatten(), weights):
            weighted_ratings += weight * self.user_item_matrix[idx]
            
        # Filter out already rated items
        mask = self.user_item_matrix[user_id] == 0
        recommendations = weighted_ratings * mask
        
        return np.argsort(recommendations)[::-1]
```

## 7. Best Practices and Common Pitfalls

### Performance Monitoring
```python
from sklearn.model_selection import learning_curve

def plot_learning_curves(X, y, k_values=[1, 3, 5, 7]):
    """Plot learning curves for different k values"""
    plt.figure(figsize=(12, 8))
    
    for k in k_values:
        train_sizes, train_scores, val_scores = learning_curve(
            KNeighborsClassifier(n_neighbors=k),
            X, y, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.plot(train_sizes, train_mean, label=f'k={k}')
        plt.fill_between(train_sizes, 
                        train_mean - train_std,
                        train_mean + train_std, 
                        alpha=0.1)
    
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title('Learning Curves for Different k Values')
    plt.legend()
```

This enhanced explanation provides:
1. Visual representations of distance metrics and their effects
2. Step-by-step algorithm visualization
3. Interactive examples for understanding parameter effects
4. Optimization techniques with visualizations
5. Real-world applications with implementation details
6. Best practices and performance monitoring tools

The mermaid diagrams and code examples can be run to generate visual insights into how k-Nearest Neighbors works in practice.
