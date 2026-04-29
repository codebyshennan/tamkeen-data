---
reading_minutes: 15
objectives:
  - "Compute Euclidean, Manhattan, and cosine distances by hand on a 2D example."
  - "Pick a metric that fits the data: Euclidean for raw geometry, Manhattan for grid-like features, cosine for direction (text and embeddings)."
  - "Always scale features before kNN with `StandardScaler` or `MinMaxScaler` so large-range columns do not dominate distance."
---
# Understanding Distance Metrics in KNN

**After this lesson:** you can explain the core ideas in “Understanding Distance Metrics in KNN” and reproduce the examples here in your own notebook or environment.

## Overview

Compares **distance metrics** (Euclidean, Manhattan, Minkowski, etc.) and why scaling features first is non-negotiable for kNN.

[Introduction](1-introduction.md) sets the voting story; [5.2 README](../README.md) places kNN among other learners.

## Helpful video

Crash Course AI: supervised learning for classical algorithms.

<iframe width="560" height="315" src="https://www.youtube.com/embed/4qVRBYAdLAo" title="Supervised Learning: Crash Course AI" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Why Distance Metrics Matter

Imagine you're trying to find similar houses to yours:

- If you only look at price, you might miss houses that are similar in size
- If you only look at size, you might miss houses in similar locations
- You need a way to combine all these factors to find truly similar houses

This is exactly what distance metrics do in KNN - they help us measure similarity in a way that makes sense for our data.

## Common Distance Metrics Explained

{% include mermaid-diagram.html src="5-ml-fundamentals/5.2-supervised-learning-1/knn/diagrams/2-distance-metrics-1.mmd" %}

*Always apply `StandardScaler` or `MinMaxScaler` before using any distance-based method — a salary column in dollars will dwarf an age column in years.*

### 1. Euclidean Distance (Straight Line Distance)

Think of this as measuring distance "as the crow flies" - the shortest possible path between two points.

**Real-World Example:**

- Measuring the straight-line distance between two cities on a map
- Finding the shortest path for a drone to fly between two points

##### Euclidean distance in $\mathbb{R}^2$

```python
# Simple Example: Distance between two points on a map
import numpy as np

def euclidean_distance(point1, point2):
    """Calculate straight-line distance between two points"""
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Example: Distance between New York (40.7, -74.0) and Boston (42.3, -71.0)
ny = np.array([40.7, -74.0])
boston = np.array([42.3, -71.0])
distance = euclidean_distance(ny, boston)
print(f"Distance between NY and Boston: {distance:.2f} units")
```

**When to Use:**

- When you want the shortest possible distance
- For continuous numerical data
- When all features are on the same scale

### 2. Manhattan Distance (City Block Distance)

This is like walking through a city grid - you can only move along the streets, not through buildings.

**Real-World Example:**

- Calculating taxi fare in Manhattan (hence the name)
- Measuring walking distance in a city with a grid layout

##### Manhattan ($L_1$) distance

```python
def manhattan_distance(point1, point2):
    """Calculate distance moving only along grid lines"""
    return np.sum(np.abs(point1 - point2))

# Example: Walking distance in a city grid
start = np.array([0, 0])  # Starting at intersection
end = np.array([3, 4])    # Destination
distance = manhattan_distance(start, end)
print(f"Walking distance: {distance} blocks")
```

**When to Use:**

- When movement is restricted to grid-like paths
- For data where diagonal movement doesn't make sense
- When features have different scales

### 3. Cosine Similarity (Angle-Based Similarity)

This measures how similar two things are based on the angle between them, ignoring their size.

**Real-World Example:**

- Comparing two documents regardless of their length
- Finding similar products based on their features

##### Cosine similarity (not a metric distance; use `1 - cosine` for distance)

```python
def cosine_similarity(point1, point2):
    """Calculate similarity based on angle between vectors"""
    dot_product = np.dot(point1, point2)
    norm1 = np.linalg.norm(point1)
    norm2 = np.linalg.norm(point2)
    return dot_product / (norm1 * norm2)

# Example: Comparing two product descriptions
product1 = np.array([1, 0, 1, 1])  # Features: [price, quality, popularity, reviews]
product2 = np.array([2, 0, 2, 2])  # Same pattern, just larger numbers
similarity = cosine_similarity(product1, product2)
print(f"Similarity between products: {similarity:.2f}")
```

**When to Use:**

- For text data
- When magnitude doesn't matter
- For high-dimensional data

## Common Mistakes to Avoid

1. **Using Euclidean Distance with Unscaled Features**
   - Problem: Features on different scales (e.g., age vs. salary) can dominate the distance
   - Solution: Always scale your features before using Euclidean distance

2. **Choosing the Wrong Metric for Your Data**
   - Problem: Using Euclidean for categorical data
   - Solution: Match the metric to your data type:
     - Continuous: Euclidean or Manhattan
     - Categorical: Hamming distance
     - Text: Cosine similarity

3. **Ignoring Feature Importance**
   - Problem: Treating all features equally when some matter more
   - Solution: Use weighted distance metrics or feature selection

## How to Choose the Right Metric

Here's a simple decision tree to help you choose:

1. **What type of data do you have?**
   - Continuous numbers → Euclidean or Manhattan
   - Categories → Hamming distance
   - Text → Cosine similarity

2. **Are your features on the same scale?**
   - Yes → Euclidean distance
   - No → Manhattan distance or scale your features

3. **Does the size of your data matter?**
   - Yes → Euclidean distance
   - No → Cosine similarity

## Practical Example: House Price Prediction

Let's say you're predicting house prices using these features:

- Number of bedrooms
- Square footage
- Distance from city center
- Year built

#### Compare metrics on scaled features (sketch—define `X`, `y`, splits in your notebook)

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try different distance metrics
metrics = ['euclidean', 'manhattan']
results = {}

for metric in metrics:
    knn = KNeighborsRegressor(n_neighbors=5, metric=metric)
    knn.fit(X_scaled, y)
    score = knn.score(X_test_scaled, y_test)
    results[metric] = score

print("Model performance with different metrics:")
for metric, score in results.items():
    print(f"{metric}: {score:.3f}")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Scale Features</span>
    </div>
    <div class="code-callout__body">
      <p><code>StandardScaler</code> normalizes each feature to zero mean and unit variance — critical for distance-based models so features with larger ranges don't dominate the distance calculation.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="9-20" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Compare Metrics</span>
    </div>
    <div class="code-callout__body">
      <p>Loop over Euclidean and Manhattan metrics, fit a <code>KNeighborsRegressor</code> with each, and collect R² scores — comparing them reveals which metric better reflects similarity in this feature space.</p>
    </div>
  </div>
</aside>
</div>

## Gotchas

- **Applying `fit_transform` to the test set** — Calling `scaler.fit_transform(X_test)` re-computes the mean and standard deviation from the test data, so test features are on a different scale than training features. Always call `scaler.fit_transform(X_train)` once and `scaler.transform(X_test)` thereafter.
- **Using cosine similarity as a drop-in distance for KNN** — Cosine similarity is bounded between -1 and 1, not 0 to ∞. Passing `metric='cosine'` to `KNeighborsClassifier` actually computes cosine distance (`1 - cosine_similarity`), which is valid, but raw cosine similarity values are not distances and will produce wrong neighbor rankings if used directly.
- **Expecting Euclidean distance to work for high-cardinality one-hot features** — One-hot encoding a categorical column with 50 values adds 50 binary dimensions. Euclidean distance treats each as equally spaced, which inflates distances for categories even if two points differ by only one category. Manhattan distance is often a better fit here.
- **Forgetting that MinMaxScaler is sensitive to outliers** — If your data contains outliers, `MinMaxScaler` compresses all non-outlier values into a tiny range, distorting distances. `StandardScaler` (z-score) is less affected; consider clipping outliers first when using MinMaxScaler.
- **Mixing scaled and unscaled features accidentally** — A common mistake when adding engineered features is to forget to include them in the scaler's input. Features added after `fit_transform` bypass scaling entirely and dominate distance calculations silently.
- **Treating Minkowski distance with p=1 and Manhattan as identical but forgetting `p` must be set explicitly** — `KNeighborsClassifier` defaults to `metric='minkowski'` with `p=2` (Euclidean). If you want Manhattan distance, you must pass `p=1` or `metric='manhattan'`; changing only `metric` to `'minkowski'` without changing `p` leaves you with Euclidean.

## Additional Resources

For more learning:

- [Scikit-learn Distance Metrics](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html)
- [Interactive Distance Metric Visualizer](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote16.html)
- [Distance Metric Cheat Sheet](https://www.kdnuggets.com/2020/08/most-popular-distance-metrics-knn.html)

Remember: The right distance metric is like choosing the right tool for the job. Just as you wouldn't use a ruler to measure the weight of an object, you shouldn't use Euclidean distance for all types of data!
