# Implementing KNN: From Scratch to Scikit-learn 💻

Let's learn how to implement KNN both from scratch (to understand the algorithm deeply) and using scikit-learn (for practical applications).

## Implementation from Scratch 🔨

### Basic KNN Classifier

```python
import numpy as np
from collections import Counter

class SimpleKNN:
    def __init__(self, k=3):
        """Initialize KNN with k neighbors"""
        self.k = k
        
    def fit(self, X, y):
        """Store training data - no actual training needed!"""
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        """Make predictions for each point in X"""
        return np.array([self._predict_single(x) for x in X])
    
    def _predict_single(self, x):
        """Predict class for a single point"""
        # Calculate distances to all training points
        distances = [np.sqrt(np.sum((x - x_train)**2)) 
                    for x_train in self.X_train]
        
        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return most common class
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Example usage
X_train = np.array([
    [1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]
])
y_train = np.array(['A', 'A', 'B', 'B', 'A', 'B'])

# Create and train model
knn = SimpleKNN(k=3)
knn.fit(X_train, y_train)

# Make prediction
new_point = np.array([3, 4])
prediction = knn.predict([new_point])
print(f"Predicted class: {prediction[0]}")
```

### Basic KNN Regressor

```python
class SimpleKNNRegressor:
    def __init__(self, k=3):
        """Initialize KNN Regressor"""
        self.k = k
        
    def fit(self, X, y):
        """Store training data"""
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        """Predict values for each point in X"""
        return np.array([self._predict_single(x) for x in X])
    
    def _predict_single(self, x):
        """Predict value for a single point"""
        # Calculate distances
        distances = [np.sqrt(np.sum((x - x_train)**2)) 
                    for x_train in self.X_train]
        
        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_values = [self.y_train[i] for i in k_indices]
        
        # Return average value
        return np.mean(k_nearest_values)

# Example usage
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 5, 4, 5])

# Create and train model
knn_reg = SimpleKNNRegressor(k=2)
knn_reg.fit(X_train, y_train)

# Make prediction
new_point = np.array([[2.5]])
prediction = knn_reg.predict(new_point)
print(f"Predicted value: {prediction[0]:.2f}")
```

## Using Scikit-learn 🛠️

### Classification Example

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset: Iris classification
from sklearn.datasets import load_iris

def iris_classification_example():
    """Complete example of KNN classification with Iris dataset"""
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform',  # or 'distance'
        metric='euclidean'
    )
    knn.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test_scaled)
    
    # Evaluate
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                              target_names=iris.target_names))
    
    return knn, scaler

# Run example
model, scaler = iris_classification_example()
```

### Regression Example

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

def house_price_prediction():
    """Example of KNN regression for house price prediction"""
    # Sample data: [size, bedrooms, age]
    X = np.array([
        [1400, 3, 10],
        [1600, 3, 8],
        [1700, 4, 15],
        [1875, 4, 5],
        [1100, 2, 20]
    ])
    
    # House prices
    y = np.array([250000, 280000, 300000, 350000, 200000])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    knn = KNeighborsRegressor(
        n_neighbors=3,
        weights='distance'  # Weight by inverse of distance
    )
    knn.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test_scaled)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: ${mse:,.2f}")
    print(f"R² Score: {r2:.3f}")
    
    return knn, scaler

# Run example
model, scaler = house_price_prediction()
```

## Best Practices for Implementation 📚

### 1. Data Preprocessing

```python
def preprocess_data(X, categorical_features=[]):
    """Preprocess data for KNN"""
    # Create pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    
    # Define transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first')
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, 
             [i for i in range(X.shape[1]) if i not in categorical_features]),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor
```

### 2. Model Selection

```python
from sklearn.model_selection import GridSearchCV

def optimize_knn(X, y, cv=5):
    """Find optimal KNN parameters"""
    # Parameter grid
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    # Create model
    knn = KNeighborsClassifier()
    
    # Grid search
    grid_search = GridSearchCV(
        knn, param_grid, cv=cv,
        scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X, y)
    
    print("Best parameters:", grid_search.best_params_)
    return grid_search.best_estimator_
```

### 3. Performance Evaluation

```python
def evaluate_knn(model, X, y, cv=5):
    """Evaluate KNN model thoroughly"""
    from sklearn.model_selection import cross_val_score
    
    # Cross-validation scores
    scores = cross_val_score(model, X, y, cv=cv)
    
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

## Common Pitfalls and Solutions ⚠️

1. **High Dimensionality**
   ```python
   from sklearn.decomposition import PCA
   
   # Reduce dimensions while preserving variance
   pca = PCA(n_components=0.95)  # Keep 95% of variance
   X_reduced = pca.fit_transform(X)
   ```

2. **Imbalanced Classes**
   ```python
   from imblearn.over_sampling import SMOTE
   
   # Balance classes using SMOTE
   smote = SMOTE(random_state=42)
   X_balanced, y_balanced = smote.fit_resample(X, y)
   ```

3. **Memory Issues**
   ```python
   from sklearn.neighbors import KNeighborsClassifier
   
   # Use ball tree algorithm for better memory efficiency
   knn = KNeighborsClassifier(
       algorithm='ball_tree',
       leaf_size=30  # Adjust for speed/memory trade-off
   )
   ```

## Next Steps 📚

Now that you can implement KNN:
1. Explore [advanced techniques](4-advanced.md)
2. Learn about [real-world applications](5-applications.md)
3. Practice with different datasets
4. Experiment with various distance metrics

Remember:
- Start simple and iterate
- Always preprocess your data
- Cross-validate your models
- Monitor computational resources
