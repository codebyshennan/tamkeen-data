# k-Nearest Neighbors (kNN) 🎯

k-Nearest Neighbors is an intuitive, instance-based learning algorithm that makes predictions based on the most similar examples in the training data. Think of it as recommending music based on what similar people enjoy - if 5 people your age living nearby love jazz, you might too!

## Mathematical Foundation 📐

### Distance Metrics

1. **Euclidean Distance** (L2 norm):
$$d(p,q) = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}$$

2. **Manhattan Distance** (L1 norm):
$$d(p,q) = \sum_{i=1}^n |p_i - q_i|$$

3. **Minkowski Distance** (Lp norm):
$$d(p,q) = \left(\sum_{i=1}^n |p_i - q_i|^p\right)^{\frac{1}{p}}$$

### Prediction Rules

For classification with $k$ neighbors:
$$\hat{y} = \text{mode}\{y_i : i \in N_k(x)\}$$

For regression with $k$ neighbors:
$$\hat{y} = \frac{1}{k}\sum_{i \in N_k(x)} y_i$$

Where $N_k(x)$ represents the indices of the $k$ nearest neighbors to point $x$.

## Algorithm Implementation 🔄

### Basic kNN Classifier
```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

class CustomKNNClassifier:
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
        
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
        
    def manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))
        
    def get_neighbors(self, x):
        # Calculate distances to all training points
        distances = []
        for i in range(len(self.X_train)):
            if self.metric == 'euclidean':
                dist = self.euclidean_distance(x, self.X_train[i])
            else:
                dist = self.manhattan_distance(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))
            
        # Sort by distance and get top k
        distances.sort(key=lambda x: x[0])
        return [d[1] for d in distances[:self.k]]
        
    def predict(self, X):
        predictions = []
        for x in X:
            neighbors = self.get_neighbors(x)
            prediction = max(set(neighbors), key=neighbors.count)
            predictions.append(prediction)
        return np.array(predictions)
```

## Visual Understanding 📊

### Decision Boundaries
```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# Create and train model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Create mesh grid
h = 0.02  # step size
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Plot decision boundary
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.title('kNN Decision Boundary (k=3)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.show()
```

## Real-World Applications 🌍

### 1. Recommendation Systems
```python
class MovieRecommender:
    def __init__(self, k=5):
        self.k = k
        
    def fit(self, user_movie_ratings):
        self.ratings = user_movie_ratings
        return self
        
    def recommend(self, user_ratings):
        """Recommend movies based on similar users"""
        # Calculate similarities
        similarities = []
        for other_ratings in self.ratings:
            # Only consider movies both users have rated
            common_movies = user_ratings.notnull() & other_ratings.notnull()
            if common_movies.sum() == 0:
                similarities.append(0)
            else:
                correlation = np.corrcoef(
                    user_ratings[common_movies],
                    other_ratings[common_movies]
                )[0,1]
                similarities.append(correlation)
                
        # Get top k similar users
        similar_users = np.argsort(similarities)[-self.k:]
        
        # Get recommendations
        recommendations = []
        for movie in user_ratings[user_ratings.isnull()].index:
            ratings = []
            for user in similar_users:
                if not np.isnan(self.ratings[user][movie]):
                    ratings.append(self.ratings[user][movie])
            if ratings:
                recommendations.append((movie, np.mean(ratings)))
                
        return sorted(recommendations, key=lambda x: x[1], reverse=True)
```

### 2. Image Classification
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

class ImageClassifier:
    def __init__(self, k=5):
        self.k = k
        self.scaler = StandardScaler()
        self.classifier = KNeighborsClassifier(
            n_neighbors=k,
            weights='distance'  # Weight by inverse of distance
        )
        
    def preprocess(self, images):
        """Flatten and scale images"""
        # Flatten 2D images to 1D
        flattened = images.reshape(images.shape[0], -1)
        # Scale pixel values
        return self.scaler.fit_transform(flattened)
        
    def fit(self, X, y):
        """Train the classifier"""
        X_processed = self.preprocess(X)
        self.classifier.fit(X_processed, y)
        return self
        
    def predict(self, X):
        """Make predictions"""
        X_processed = self.preprocess(X)
        return self.classifier.predict(X_processed)
```

## Parameter Selection and Optimization 🔧

### 1. Finding Optimal k
```python
from sklearn.model_selection import GridSearchCV

def find_best_k(X, y, k_range=range(1, 31, 2)):
    """Find optimal k using cross-validation"""
    param_grid = {'n_neighbors': k_range}
    grid_search = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X, y)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, grid_search.cv_results_['mean_test_score'])
    plt.fill_between(
        k_range,
        grid_search.cv_results_['mean_test_score'] - 
        grid_search.cv_results_['std_test_score'],
        grid_search.cv_results_['mean_test_score'] + 
        grid_search.cv_results_['std_test_score'],
        alpha=0.2
    )
    plt.xlabel('k (number of neighbors)')
    plt.ylabel('Cross-validation accuracy')
    plt.title('Finding Optimal k')
    plt.grid(True)
    plt.show()
    
    return grid_search.best_params_['n_neighbors']
```

### 2. Distance Metric Selection
```python
def compare_distance_metrics(X, y):
    """Compare different distance metrics"""
    metrics = ['euclidean', 'manhattan', 'minkowski']
    scores = {}
    
    for metric in metrics:
        knn = KNeighborsClassifier(metric=metric)
        cv_scores = cross_val_score(knn, X, y, cv=5)
        scores[metric] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
        
    return scores
```

## Handling High Dimensions 🌌

### 1. Dimensionality Reduction
```python
from sklearn.decomposition import PCA

def reduce_dimensions(X, n_components=0.95):
    """Reduce dimensions while preserving variance"""
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    
    # Plot explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Analysis')
    plt.grid(True)
    plt.show()
    
    return X_reduced, pca
```

### 2. Feature Selection
```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def select_features(X, y, k=10):
    """Select top k features using mutual information"""
    selector = SelectKBest(
        score_func=mutual_info_classif,
        k=k
    )
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature indices
    selected_features = selector.get_support()
    
    return X_selected, selected_features
```

## Best Practices and Optimization 💡

### 1. Data Preprocessing
```python
def preprocess_for_knn(X):
    """Prepare data for kNN"""
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_scaled)
    
    return X_imputed, scaler, imputer
```

### 2. Performance Optimization
```python
from sklearn.neighbors import KDTree

class OptimizedKNN:
    def __init__(self, k=5):
        self.k = k
        self.tree = None
        
    def fit(self, X, y):
        """Build KD-tree for efficient neighbor search"""
        self.tree = KDTree(X)
        self.y_train = y
        return self
        
    def predict(self, X):
        """Make predictions using KD-tree"""
        distances, indices = self.tree.query(X, k=self.k)
        predictions = []
        
        for idx_list in indices:
            neighbors = self.y_train[idx_list]
            pred = np.bincount(neighbors).argmax()
            predictions.append(pred)
            
        return np.array(predictions)
```

## Common Pitfalls and Solutions ⚠️

1. **Curse of Dimensionality**
   - Use dimensionality reduction
   - Feature selection
   - Increase k with dimension

2. **Computational Complexity**
   - Use KD-trees or Ball-trees
   - Sample large datasets
   - Parallel processing

3. **Imbalanced Data**
   - Use weighted voting
   - Adjust neighbor weights
   - Balance training data

## Next Steps 📚

Now that you understand k-Nearest Neighbors, let's explore [Support Vector Machines](./svm.md) to learn about maximum margin classification!
