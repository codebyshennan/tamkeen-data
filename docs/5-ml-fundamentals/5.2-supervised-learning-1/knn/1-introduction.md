---
reading_minutes: 15
objectives:
  - >-
    Describe kNN as a lazy learner: store training data, defer all work to
    prediction time.
  - >-
    Choose between classification (majority vote) and regression (neighbour
    average) for a given problem.
  - >-
    Identify the three knobs that change kNN behaviour: `k`, distance metric,
    and feature scaling.
---

# Introduction to k-Nearest Neighbors (KNN)

**After this lesson:** you can explain Introduction to k-Nearest Neighbors (KNN) and try the examples in your own notebook.

## Overview

**kNN** is a **lazy** learner: it stores training points and, at prediction time, finds the $k$ closest neighbors by a distance metric, then votes (classification) or averages (regression). **Prerequisites:** [5.2 README](../); [distance metrics](2-distance-metrics.md) and scaling matter a lot in practice.

Welcome to your journey into k-Nearest Neighbors (KNN)! This algorithm is one of the most intuitive ways to start learning about machine learning. Think of it as your friendly neighborhood algorithm that makes decisions based on what's closest to it.

![KNN Decision Boundary](../../../../.gitbook/assets/knn_decision_boundary.png) _Figure: KNN Decision Boundary showing how the algorithm classifies different regions_

## What is KNN?

> **k-Nearest Neighbors** is like having a group of friends who help you make decisions. The algorithm looks at the "k" closest examples in your data and makes predictions based on their characteristics.

### Real-World Analogies

1. **Movie Recommendations**
   * Imagine you're trying to decide what movie to watch
   * You ask your 5 closest friends (k=5) for recommendations
   * If 3 friends recommend action movies and 2 recommend comedies
   * You'll likely choose an action movie because it got more "votes"
2. **Restaurant Selection**
   * You're in a new city looking for dinner
   * You ask the 3 nearest people (k=3) for recommendations
   * 2 suggest Italian, 1 suggests Chinese
   * You choose Italian because it's the majority vote
3. **House Price Prediction**
   * You want to estimate your house's value
   * You look at the sale prices of the 5 most similar houses in your neighborhood
   * The average of these prices gives you a good estimate

## Why KNN Matters

KNN is important because:

* It's easy to understand and implement
* It works well for both classification and regression problems
* It doesn't make assumptions about your data
* It's great for exploring patterns in your data
* It's perfect for beginners to understand how machine learning works

## How Does KNN Work?

break it down into simple steps:

1. **Find Neighbors**
   * When you have a new data point
   * KNN looks for the "k" most similar points in your dataset
   * Similarity is measured using distance metrics (like how far apart two points are)
2. **Make a Decision**
   * For classification: Take a vote among the neighbors
   * For regression: Take the average of the neighbors' values

_At prediction time KNN does all the work, it never builds an explicit model. That's why it's called a **lazy** learner._

#### Minimal `KNeighborsClassifier` usage

```python
# Simple KNN Example
from sklearn.neighbors import KNeighborsClassifier

# Create a KNN classifier with 3 neighbors
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model (KNN just stores the data)
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)
```

## Common Mistakes to Avoid

1. **Choosing the Wrong k Value**
   * Too small (k=1): Too sensitive to noise
   * Too large: May include points from other classes
   * Solution: Try different values and use cross-validation
2. **Forgetting to Scale Features**
   * Features on different scales can distort distances
   * Solution: Always scale your features before using KNN
3. **Using KNN with Large Datasets**
   * KNN can be slow with big datasets
   * Solution: Consider using approximate nearest neighbors or other algorithms

## When to Use KNN?

### Perfect For

* Small to medium datasets
* Problems where local patterns matter
* When you need interpretable results
* When you want to understand your data better

### Not Great For

* Very large datasets (slow predictions)
* High-dimensional data (curse of dimensionality)
* When features are on different scales
* When memory is limited

## Key Decisions in KNN

1. **Choosing k**
   * Start with k=5 and experiment
   * Use cross-validation to find the best k
   * Consider using odd numbers for classification to avoid ties
2. **Distance Metric**
   * Euclidean: Good for continuous features
   * Manhattan: Better for categorical features
   * Cosine: Great for text data
3. **Weighting**
   * Uniform: All neighbors have equal weight
   * Distance-based: Closer neighbors have more influence

## Next Steps

In the following sections, we'll explore:

1. [Distance Metrics](2-distance-metrics.md) - How to measure similarity
2. [Implementation](3-implementation.md) - How to code KNN
3. [Advanced Techniques](4-advanced.md) - Making KNN work better
4. [Applications](5-applications.md) - Real-world uses of KNN

## Gotchas

* **Assuming KNN "trains" like other models**: KNN stores the entire training set and does all computation at prediction time. This means `fit` is instantaneous but `predict` is slow, and memory usage scales linearly with training data size. With 1 million rows, every prediction scans all 1 million points.
* **Skipping feature scaling**: If one feature spans 0-100,000 (e.g., salary) and another spans 0-5 (e.g., rating), distance is almost entirely determined by salary. KNN will silently ignore the rating feature, producing misleading results without any error.
* **Using an even k with binary classification**: An even k can produce exact ties (e.g., 2 neighbors in each class), and scikit-learn breaks ties by choosing the lower class index rather than raising an error. Use odd k values for binary problems to avoid non-deterministic-looking results.
* **Testing KNN on the same data used to find k**: If you loop over k values and pick the best on the test set, you've leaked the test set into model selection. Use cross-validation on the training set to choose k, then evaluate on the held-out test set once.
* **Ignoring the curse of dimensionality**: With many features, all pairwise distances tend toward the same value, making "nearest" neighbors meaningless. KNN typically degrades above \~20 relevant features unless you apply dimensionality reduction first.
* **Treating KNN as memory-free after training**: Unlike tree or linear models, the entire `X_train` array lives in memory for the lifetime of the fitted object. Storing a KNN model does not discard training data, so deployment memory costs are much higher than for parametric models.

## Additional Resources

For further learning:

* [Scikit-learn KNN Documentation](https://scikit-learn.org/stable/modules/neighbors.html)
* [KNN Visualization Tool](https://www.cs.waikato.ac.nz/ml/weka/)
* [Interactive KNN Demo](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote16.html)

Remember: KNN is like having a group of friends who help you make decisions. The more you understand about your data, the better your "friends" can help you!
