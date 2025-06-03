# Implementing KNN: A Step-by-Step Guide

Welcome to the practical side of KNN! In this section, we'll learn how to implement KNN both from scratch (to understand how it works) and using scikit-learn (for real-world applications).

![Effect of Different k Values](assets/knn_different_k.png)
*Figure: How different values of k affect the decision boundary in KNN*

## Understanding k in KNN

The parameter k in KNN (k-Nearest Neighbors) is a crucial hyperparameter that determines how many neighboring data points to consider when making a prediction. Here's what you need to know about k:

- **What is k?**: k is the number of nearest neighbors that the algorithm considers when making a prediction
- **How it works**:
  - For a new data point, KNN finds the k closest points in the training data
  - The algorithm then takes a "majority vote" among these k neighbors
  - The most common class among these k neighbors becomes the prediction
- **Impact of k**:
  - Small k (e.g., k=1): More sensitive to noise, captures local patterns
  - Large k: More stable but might include irrelevant points
  - Rule of thumb: Start with k = √n (where n is number of training samples)

Think of k like asking for advice:

- k=1 is like asking only your closest friend
- k=5 is like asking your 5 closest friends
- k=20 is like asking a larger group of friends

## Why Implementation Matters

Understanding how to implement KNN is crucial because:

- It helps you understand how the algorithm works under the hood
- You can customize it for your specific needs
- You'll be better at debugging when things go wrong
- You can optimize it for your particular use case

## Implementation from Scratch

Let's build a simple KNN classifier step by step. Think of it like building a recommendation system that asks your closest friends for advice.

### Step 1: Create the Basic Structure

```python
import numpy as np
from collections import Counter

class SimpleKNN:
    def __init__(self, k=3):
        """Initialize with k neighbors (default: 3)"""
        self.k = k
        
    def fit(self, X, y):
        """Store the training data - KNN doesn't actually train!"""
        self.X_train = X
        self.y_train = y
```

**What's happening here:**

- We create a class called `SimpleKNN`
- The `__init__` method sets up how many neighbors (k) we want to consider
- The `fit` method just stores our training data (unlike other algorithms, KNN doesn't need training!)

### Step 2: Add Prediction Logic

```python
    def predict(self, X):
        """Make predictions for new data points"""
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
```

**Breaking it down:**

1. `predict` handles multiple points at once
2. `_predict_single` works on one point at a time:
   - Calculates distances to all training points
   - Finds the k closest points
   - Returns the most common class among them

### Step 3: Try it Out

```python
# Example: Movie Genre Classification
# Features: [Action Score, Romance Score]
X_train = np.array([
    [8, 2],  # Action movie
    [7, 3],  # Action movie
    [2, 8],  # Romance movie
    [3, 7],  # Romance movie
    [1, 9],  # Romance movie
    [9, 1]   # Action movie
])
y_train = np.array(['Action', 'Action', 'Romance', 'Romance', 'Romance', 'Action'])

# Create and train model
knn = SimpleKNN(k=3)
knn.fit(X_train, y_train)

# Predict a new movie
new_movie = np.array([4, 6])  # Mix of action and romance
prediction = knn.predict([new_movie])
print(f"Predicted genre: {prediction[0]}")
```

## Using Scikit-learn

While implementing from scratch is educational, scikit-learn provides a robust, optimized version of KNN. Let's see how to use it for a real-world problem.

### Example: Iris Flower Classification

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

def classify_iris_flowers():
    """Complete example of classifying iris flowers"""
    # Load the famous Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features (important for KNN!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    knn = KNeighborsClassifier(
        n_neighbors=5,          # Number of neighbors to consider
        weights='uniform',      # All neighbors have equal weight
        metric='euclidean'      # Distance metric to use
    )
    knn.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test_scaled)
    
    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred,
                              target_names=iris.target_names))
    
    return knn, scaler

# Run the example
model, scaler = classify_iris_flowers()
```

## Common Mistakes to Avoid

1. **Forgetting to Scale Features**

   ```python
   #  Wrong way
   knn = KNeighborsClassifier()
   knn.fit(X_train, y_train)  # Features not scaled
   
   #  Right way
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   knn.fit(X_train_scaled, y_train)
   ```

2. **Choosing the Wrong k Value**

   ```python
   #  Using k=1 (too sensitive to noise)
   knn = KNeighborsClassifier(n_neighbors=1)
   
   #  Try different values and use cross-validation
   from sklearn.model_selection import GridSearchCV
   param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
   grid_search = GridSearchCV(knn, param_grid, cv=5)
   grid_search.fit(X_train_scaled, y_train)
   ```

3. **Not Handling Categorical Features**

   ```python
   #  Using categorical features directly
   knn.fit(X_with_categories, y)
   
   #  Encode categorical features first
   from sklearn.preprocessing import OneHotEncoder
   encoder = OneHotEncoder()
   X_encoded = encoder.fit_transform(X_with_categories)
   knn.fit(X_encoded, y)
   ```

## Best Practices

1. **Always Scale Your Features**

   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **Use Cross-Validation**

   ```python
   from sklearn.model_selection import cross_val_score
   scores = cross_val_score(knn, X_scaled, y, cv=5)
   print(f"Average accuracy: {scores.mean():.3f}")
   ```

3. **Optimize Hyperparameters**

   ```python
   from sklearn.model_selection import GridSearchCV
   
   param_grid = {
       'n_neighbors': [3, 5, 7, 9, 11],
       'weights': ['uniform', 'distance'],
       'metric': ['euclidean', 'manhattan']
   }
   
   grid_search = GridSearchCV(knn, param_grid, cv=5)
   grid_search.fit(X_scaled, y)
   print(f"Best parameters: {grid_search.best_params_}")
   ```

## Detailed Implementation Guide

### Understanding Common Mistakes in Depth

1. **Feature Scaling: Why It's Critical**
   - **The Problem**: KNN is distance-based, making it sensitive to feature scales
   - **Real-world Impact**:
     - Features with larger scales (e.g., income: 0-1000000) dominate distance calculations
     - Features with smaller scales (e.g., age: 0-100) become less influential
   - **Solution Details**:

     ```python
     # 1. Create the scaler
     scaler = StandardScaler()
     
     # 2. Fit and transform training data
     X_train_scaled = scaler.fit_transform(X_train)
     
     # 3. Transform test data (using same scaling as training)
     X_test_scaled = scaler.transform(X_test)
     ```

   - **Why StandardScaler Works**:
     - Transforms features to have mean = 0 and standard deviation = 1
     - Ensures all features contribute equally to distance calculations
     - Makes the model more robust and interpretable

2. **K Value Selection: Finding the Sweet Spot**
   - **Impact of Different k Values**:
     - Too small (k=1):
       - Pros: Captures local patterns well
       - Cons: Highly sensitive to noise, prone to overfitting
     - Too large:
       - Pros: More stable predictions
       - Cons: May include irrelevant points from other classes
   - **Optimal Selection Strategy**:
     - Start with k = √n (where n is number of training samples)
     - Use cross-validation to evaluate different k values
     - Consider the balance between bias and variance
   - **Implementation with GridSearchCV**:

     ```python
     from sklearn.model_selection import GridSearchCV
     
     # Define parameter grid
     param_grid = {
         'n_neighbors': [3, 5, 7, 9, 11],
         'weights': ['uniform', 'distance'],
         'metric': ['euclidean', 'manhattan']
     }
     
     # Create and run grid search
     grid_search = GridSearchCV(
         KNeighborsClassifier(),
         param_grid,
         cv=5,  # 5-fold cross-validation
         scoring='accuracy',
         n_jobs=-1  # Use all available CPU cores
     )
     ```

3. **Categorical Feature Handling: Beyond One-Hot Encoding**
   - **Why It Matters**:
     - KNN requires numerical features for distance calculations
     - Categorical variables need proper encoding to preserve their meaning
   - **Encoding Strategies**:
     - **One-Hot Encoding**: For nominal categories (no inherent order)

       ```python
       from sklearn.preprocessing import OneHotEncoder
       encoder = OneHotEncoder(sparse=False)
       X_encoded = encoder.fit_transform(X_categorical)
       ```

     - **Label Encoding**: For ordinal categories (has inherent order)

       ```python
       from sklearn.preprocessing import LabelEncoder
       encoder = LabelEncoder()
       X_encoded = encoder.fit_transform(X_ordinal)
       ```

   - **Best Practices**:
     - Always use One-Hot Encoding for nominal categories
     - Consider feature interactions after encoding
     - Handle missing values before encoding

### Advanced Best Practices

1. **Cross-Validation: Beyond Basic Implementation**
   - **Purpose and Benefits**:
     - More reliable performance estimation
     - Better use of limited data
     - Early detection of overfitting
   - **Implementation with Detailed Metrics**:

     ```python
     from sklearn.model_selection import cross_val_score
     from sklearn.metrics import make_scorer, accuracy_score, f1_score
     
     # Define multiple scoring metrics
     scoring = {
         'accuracy': 'accuracy',
         'f1': 'f1_weighted'
     }
     
     # Perform cross-validation with multiple metrics
     scores = cross_validate(
         knn, 
         X_scaled, 
         y,
         cv=5,
         scoring=scoring,
         return_train_score=True
     )
     
     # Print detailed results
     print(f"Training Accuracy: {scores['train_accuracy'].mean():.3f} (+/- {scores['train_accuracy'].std() * 2:.3f})")
     print(f"Validation Accuracy: {scores['test_accuracy'].mean():.3f} (+/- {scores['test_accuracy'].std() * 2:.3f})")
     ```

2. **Hyperparameter Optimization: A Systematic Approach**
   - **Key Parameters to Tune**:
     - `n_neighbors`: Number of neighbors (k)
     - `weights`: How to weight the neighbors
       - 'uniform': All neighbors have equal weight
       - 'distance': Weight by inverse of distance
     - `metric`: Distance metric to use
       - 'euclidean': Standard straight-line distance
       - 'manhattan': City-block distance
       - 'minkowski': Generalization of both
   - **Comprehensive Grid Search**:

     ```python
     from sklearn.model_selection import GridSearchCV
     
     # Define extensive parameter grid
     param_grid = {
         'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
         'weights': ['uniform', 'distance'],
         'metric': ['euclidean', 'manhattan', 'minkowski'],
         'p': [1, 2, 3]  # For Minkowski distance
     }
     
     # Create and run grid search with parallel processing
     grid_search = GridSearchCV(
         KNeighborsClassifier(),
         param_grid,
         cv=5,
         scoring='accuracy',
         n_jobs=-1,
         verbose=1
     )
     
     # Fit and get best parameters
     grid_search.fit(X_scaled, y)
     print(f"Best parameters: {grid_search.best_params_}")
     print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
     ```

3. **Model Evaluation and Monitoring**
   - **Performance Metrics**:
     - Accuracy: Overall correctness
     - Precision: Accuracy of positive predictions
     - Recall: Ability to find all positive cases
     - F1-score: Harmonic mean of precision and recall
   - **Implementation**:

     ```python
     from sklearn.metrics import classification_report, confusion_matrix
     
     # Get predictions
     y_pred = knn.predict(X_test_scaled)
     
     # Print detailed classification report
     print(classification_report(y_test, y_pred))
     
     # Create confusion matrix
     cm = confusion_matrix(y_test, y_pred)
     print("Confusion Matrix:")
     print(cm)
     ```

Remember: Successful KNN implementation requires careful consideration of:

- Data preprocessing and scaling
- Appropriate k value selection
- Proper handling of categorical variables
- Systematic hyperparameter optimization
- Comprehensive model evaluation

## Additional Resources

For more learning:

- [Scikit-learn KNN Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [KNN Visualization Tool](https://www.cs.waikato.ac.nz/ml/weka/)
- [Interactive KNN Demo](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote16.html)

Remember: The key to successful KNN implementation is understanding your data and choosing the right parameters. Don't be afraid to experiment and try different approaches!
