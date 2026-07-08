---
reading_minutes: 25
objectives:
  - >-
    Implement kNN from scratch (distance + top-k vote) to see what
    `KNeighborsClassifier` does internally.
  - >-
    Use `KNeighborsClassifier` and `KNeighborsRegressor` end-to-end with scaling
    and a held-out split.
  - >-
    Pick `k` from a validation curve, prefer odd values for binary
    classification, and recognise the boundary-smoothness vs noise tradeoff.
---

# Implementing KNN: A Step-by-Step Guide

**After this lesson:** you can explain Implementing KNN: A Step-by-Step Guide and try the examples in your own notebook.

## Overview

**`KNeighborsClassifier` / `Regressor`** in scikit-learn: `n_neighbors`, weights, and brute vs KD-tree ball queries at a high level.

## Understanding k in KNN

The parameter k in KNN (k-Nearest Neighbors) is a important hyperparameter that determines how many neighboring data points to consider when making a prediction. about k:

* **What is k?**: k is the number of nearest neighbors that the algorithm considers when making a prediction
* **How it works**:
  * For a new data point, KNN finds the k closest points in the training data
  * The algorithm then takes a "majority vote" among these k neighbors
  * The most common class among these k neighbors becomes the prediction
* **Impact of k**:
  * Small k (e.g., k=1): More sensitive to noise, captures local patterns
  * Large k: More stable but might include irrelevant points
  * Rule of thumb: Start with k = √n (where n is number of training samples)

Think of k like asking for advice:

* k=1 is like asking only your closest friend
* k=5 is like asking your 5 closest friends
* k=20 is like asking a larger group of friends

_The `√n` rule is a starting point. Always use cross-validation to find the optimal k for your dataset._

## Why Implementation Matters

Understanding how to implement KNN is important because:

* It helps you understand how the algorithm works under the hood
* You can customize it for your specific needs
* You'll be better at debugging when things go wrong
* You can optimize it for your particular use case

## Implementation from Scratch

Build a simple KNN classifier step by step. Think of it like building a recommendation system that asks your closest friends for advice.

### Step 1: Create the Basic Structure

#### SimpleKNN class skeleton

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

* We create a class called `SimpleKNN`
* The **`init`** method sets up how many neighbors (k) we want to consider
* The `fit` method just stores our training data (unlike other algorithms, KNN doesn't need training!)

### Step 2: Add Prediction Logic

#### `SimpleKNN` prediction methods

Batch Predict

`predict` maps `_predict_single` over every row of `X` and packs the results into a NumPy array, giving the same interface as scikit-learn estimators.

Distance and Majority Vote

Euclidean distance is computed to every training point; `np.argsort` ranks them and a slice picks the k smallest; `Counter.most_common(1)` returns the plurality class among those k neighbours.

**Breaking it down:**

1. `predict` handles multiple points at once
2. `_predict_single` works on one point at a time:
   * Calculates distances to all training points
   * Finds the k closest points
   * Returns the most common class among them

### Step 3: Try it Out

#### Demo: synthetic movie genres with `SimpleKNN(k=3)`

Training Data

Six labelled movies are represented as \[action score, romance score] pairs; the two-feature space makes it easy to visualise how the nearest-neighbour boundary separates Action from Romance points.

Fit and Predict

`SimpleKNN(k=3)` is fitted by storing training data, then queried with a mixed-genre point \[4, 6]; the three nearest neighbours vote on the predicted label.

## Using Scikit-learn

While implementing from scratch is educational, scikit-learn provides a reliable, optimized version of KNN. Look at how to use it for a real-world problem.

### Example: Iris Flower Classification

#### Iris pipeline: split, scale, `KNeighborsClassifier`, metrics

Scale before KNN, always

KNN computes distances between data points. If one feature spans 0-1000 and another spans 0-1, distances are dominated by the large-scale feature. `StandardScaler` puts all features on the same scale so every dimension contributes equally.

KNN parameters

`n_neighbors=5` considers the 5 closest points for a majority vote. `weights='uniform'` treats all neighbors equally; try `'distance'` to weight closer points more. `metric='euclidean'` is the straight-line distance, other options include `'manhattan'`.

Classification report

`classification_report` shows per-class precision, recall, and F1, more informative than a single accuracy number, especially when classes are imbalanced. `target_names` replaces numeric labels with human-readable names.

```
Accuracy: 1.0

Detailed Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
```

## Common Mistakes to Avoid

1.  **Forgetting to Scale Features**

    #### Wrong vs right: scale features before `KNeighborsClassifier`

    ```python
    #  Wrong way
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)  # Features not scaled

    #  Right way
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    knn.fit(X_train_scaled, y_train)
    ```
2.  **Choosing the Wrong k Value**

    #### Grid search `n_neighbors` instead of fixing `k=1`

    ```python
    #  Using k=1 (too sensitive to noise)
    knn = KNeighborsClassifier(n_neighbors=1)

    #  Try different values and use cross-validation
    from sklearn.model_selection import GridSearchCV
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
    grid_search = GridSearchCV(knn, param_grid, cv=5)
    grid_search.fit(X_train_scaled, y_train)
    ```
3.  **Not Handling Categorical Features**

    #### Encode categories before distance-based fitting

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

1.  **Always Scale Your Features**

    #### Apply `StandardScaler` to the full feature matrix

    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ```
2.  **Use Cross-Validation**

    #### Mean CV accuracy with `cross_val_score`

    ```python
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(knn, X_scaled, y, cv=5)
    print(f"Average accuracy: {scores.mean():.3f}")
    ```
3.  **Optimize Hyperparameters**

    #### Joint grid over `n_neighbors`, `weights`, and `metric`

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
   * **The Problem**: KNN is distance-based, making it sensitive to feature scales
   * **Real-world Impact**:
     * Features with larger scales (e.g., income: 0-1000000) dominate distance calculations
     * Features with smaller scales (e.g., age: 0-100) become less influential
   *   **Solution Details**:

       #### Fit scaler on train only; transform test with the same stats

       ```python
       # 1. Create the scaler
       scaler = StandardScaler()

       # 2. Fit and transform training data
       X_train_scaled = scaler.fit_transform(X_train)

       # 3. Transform test data (using same scaling as training)
       X_test_scaled = scaler.transform(X_test)
       ```
   * **Why StandardScaler Works**:
     * Transforms features to have mean = 0 and standard deviation = 1
     * Ensures all features contribute equally to distance calculations
     * Makes the model more reliable and interpretable
2. **K Value Selection: Finding the Sweet Spot**
   * **Impact of Different k Values**:
     * Too small (k=1):
       * Pros: Captures local patterns well
       * Cons: Highly sensitive to noise, prone to overfitting
     * Too large:
       * Pros: More stable predictions
       * Cons: May include irrelevant points from other classes
   * **Optimal Selection Strategy**:
     * Start with k = √n (where n is number of training samples)
     * Use cross-validation to evaluate different k values
     * Consider the balance between bias and variance
   *   **Implementation with GridSearchCV**:

       #### `GridSearchCV` setup for KNN (parallel, accuracy scoring)

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
   * **Why It Matters**:
     * KNN requires numerical features for distance calculations
     * Categorical variables need proper encoding to preserve their meaning
   *   **Encoding Strategies**:

       * **One-Hot Encoding**: For nominal categories (no inherent order)

       #### One-hot encode nominal columns (`sparse=False`)

       ```python
       from sklearn.preprocessing import OneHotEncoder
       encoder = OneHotEncoder(sparse=False)
       X_encoded = encoder.fit_transform(X_categorical)
       ```

       *   **Label Encoding**: For ordinal categories (has inherent order)

           #### Integer encode ordered categories

           ```python
           from sklearn.preprocessing import LabelEncoder
           encoder = LabelEncoder()
           X_encoded = encoder.fit_transform(X_ordinal)
           ```
   * **Best Practices**:
     * Always use One-Hot Encoding for nominal categories
     * Consider feature interactions after encoding
     * Handle missing values before encoding

### Advanced Best Practices

1. **Cross-Validation: Beyond Basic Implementation**
   * **Purpose and Benefits**:
     * More reliable performance estimation
     * Better use of limited data
     * Early detection of overfitting
   *   **Implementation with Detailed Metrics**:

       #### Multi-metric `cross_validate` with train scores

       ```python
       from sklearn.model_selection import cross_validate

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
   * **Key Parameters to Tune**:
     * `n_neighbors`: Number of neighbors (k)
     * `weights`: How to weight the neighbors
       * 'uniform': All neighbors have equal weight
       * 'distance': Weight by inverse of distance
     * `metric`: Distance metric to use
       * 'euclidean': Standard straight-line distance
       * 'manhattan': City-block distance
       * 'minkowski': Generalization of both
   *   **Comprehensive Grid Search**:

       #### Wide grid: `n_neighbors`, weights, Minkowski `metric` and `p`

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
   * **Performance Metrics**:
     * Accuracy: Overall correctness
     * Precision: Accuracy of positive predictions
     * Recall: Ability to find all positive cases
     * F1-score: Harmonic mean of precision and recall
   *   **Implementation**:

       #### Classification report and confusion matrix on held-out data

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

* Data preprocessing and scaling
* Appropriate k value selection
* Proper handling of categorical variables
* Systematic hyperparameter optimization
* Comprehensive model evaluation

## Gotchas

* **Calling `scaler.fit_transform` on `X_test`**: A very common bug in implementation. The scaler must be fitted on `X_train` only; applying `fit_transform` to `X_test` produces a differently scaled test set, invalidating any accuracy metric you compute afterward.
* **Using `sparse=False` with `OneHotEncoder` in newer scikit-learn**: The `sparse` parameter was renamed to `sparse_output` in scikit-learn 1.2. Code using `sparse=False` raises a `TypeError` on newer versions; use `sparse_output=False` or let it default and call `.toarray()` on the result.
* **Selecting k on the test set instead of via cross-validation**: The `GridSearchCV` examples use `fit(X_train_scaled, y_train)`, but if you check `best_score_` and then also evaluate on the same `X_test`, the test score is no longer a fair generalization estimate. The test set must be touched only after all parameter selection is final.
* **Forgetting to pass `target_names` to `classification_report`**: Without `target_names`, the report prints numeric class indices (0, 1, 2…) instead of human-readable labels. On imbalanced datasets, misreading which index maps to which class leads to interpreting the wrong class's precision/recall.
* **Passing `X_with_categories` (strings) directly to `KNeighborsClassifier`**: scikit-learn's KNN cannot compute distances on string features and will raise a `ValueError`. You must encode categorical columns first; skipping this step gives a clear error at `fit` time, but using `LabelEncoder` on nominal (unordered) categories creates false ordinal relationships.
* **Comparing `GridSearchCV` scores from differently scaled data**: If you run one grid search on scaled data and another on raw data to compare, the `best_score_` values are not comparable. Always apply the same preprocessing pipeline inside `GridSearchCV` so all folds use the same feature distribution.

## Additional Resources

For more learning:

* [Scikit-learn KNN Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
* [KNN Visualization Tool](https://www.cs.waikato.ac.nz/ml/weka/)
* [Interactive KNN Demo](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote16.html)

Remember: The key to successful KNN implementation is understanding your data and choosing the right parameters. Don't be afraid to experiment and try different approaches!
