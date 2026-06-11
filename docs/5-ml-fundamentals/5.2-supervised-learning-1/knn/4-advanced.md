---
reading_minutes: 20
objectives:
  - "Switch to `weights='distance'` so closer neighbours count more when class densities vary."
  - "Apply PCA before kNN when `p` is large to fight the curse of dimensionality."
  - "Tune `n_neighbors` with `GridSearchCV` and stratified k-fold to avoid noisy single-split estimates."
  - "Trade exact for fast neighbour search using `algorithm='kd_tree'` or `'ball_tree'` on larger datasets."
---
# Advanced KNN Techniques: Taking Your Skills to the Next Level

**After this lesson:** you can explain the core ideas in “Advanced KNN Techniques: Taking Your Skills to the Next Level” and reproduce the examples here in your own notebook or environment.

## Overview

Touches **curse of dimensionality**, approximate neighbors, and practical tuning when $p$ is large or classes are imbalanced.


## Why Advanced Techniques Matter

Advanced KNN techniques help you:

- Handle complex real-world data better
- Improve model accuracy
- Make predictions faster
- Deal with special cases like imbalanced data

## 1. Weighted KNN: Giving More Importance to Closer Neighbors

Think of weighted KNN like asking your friends for movie recommendations:

- Your best friend's opinion matters more than a casual acquaintance
- The closer someone is to you, the more you trust their recommendation

### How Weighted KNN Works

#### `weights='distance'` on synthetic genre features

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Create a weighted KNN model
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance'  # This makes closer neighbors count more
)

# Example: Movie Recommendations
# Features: [Action Score, Romance Score, Comedy Score]
X_train = np.array([
    [8, 2, 1],  # Action movie
    [7, 3, 2],  # Action movie
    [2, 8, 1],  # Romance movie
    [3, 7, 2],  # Romance movie
    [1, 1, 9],  # Comedy movie
    [2, 2, 8]   # Comedy movie
])
y_train = np.array(['Action', 'Action', 'Romance', 'Romance', 'Comedy', 'Comedy'])

# Train the model
knn.fit(X_train, y_train)

# Predict a new movie
new_movie = np.array([4, 4, 3])  # Mix of all genres
prediction = knn.predict([new_movie])
print(f"Predicted genre: {prediction[0]}")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-22" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Weighted KNN Setup</span>
    </div>
    <div class="code-callout__body">
      <p>Six movies labeled by genre with three score features; <code>weights='distance'</code> makes nearer neighbors cast stronger votes than distant ones — helpful when one cluster is much closer to the query.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="24-29" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Predict Genre</span>
    </div>
    <div class="code-callout__body">
      <p>The new movie with balanced genre scores is passed to <code>predict</code>; the weighted vote among the 5 nearest neighbors determines the output genre.</p>
    </div>
  </div>
</aside>
</div>

```
Predicted genre: Action
```

## 2. Dimensionality Reduction: Making Complex Data Simpler

Sometimes your data has too many features, making KNN slow and less accurate. Dimensionality reduction helps by:

- Reducing the number of features
- Keeping the most important information
- Making visualization easier

### Using PCA (Principal Component Analysis)

#### PCA to 2D and scatter-plot Iris by class

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_data(X, y):
    """Reduce dimensions and create a scatter plot"""
    # Reduce to 2 dimensions for visualization
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis')
    plt.colorbar(scatter)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Data Visualization After PCA')
    plt.show()

# Example: Iris Dataset
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
visualize_data(X, y)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-17" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">PCA Reduction</span>
    </div>
    <div class="code-callout__body">
      <p><code>PCA(n_components=2).fit_transform</code> compresses all four Iris features into two principal components that capture the most variance — enabling a 2D scatter plot of a 4D dataset.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="19-23" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Scatter with Colorbar</span>
    </div>
    <div class="code-callout__body">
      <p>Load Iris and call the helper; <code>c=y</code> colors each point by class, and the colorbar maps numeric label to color — clear cluster separation indicates the first two PCs are informative.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/4-advanced_fig_1.png" alt="4-advanced" />
<figcaption>Figure 1: Data Visualization After PCA</figcaption>
</figure>


## 3. Finding the Best k Value

Choosing the right number of neighbors (k) is crucial. Too few can lead to noise, too many can blur boundaries.

### Cross-Validation for k Selection

#### Sweep `k` with 5-fold CV and plot accuracy vs `k`

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def find_best_k(X, y, max_k=20):
    """Find the best k value using cross-validation"""
    k_values = range(1, max_k + 1)
    scores = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, X, y, cv=5).mean()
        scores.append(score)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, scores, 'o-')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Cross-validation Accuracy')
    plt.title('Finding the Best k Value')
    plt.grid(True)
    plt.show()

    best_k = k_values[np.argmax(scores)]
    print(f"Best k value: {best_k}")
    return best_k

# Example usage
best_k = find_best_k(X, y)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-4" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports</span>
    </div>
    <div class="code-callout__body">
      <p>NumPy for argmax, <code>KNeighborsClassifier</code> for fitting, <code>cross_val_score</code> for unbiased k-selection, and matplotlib for the curve plot.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="6-24" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CV Sweep</span>
    </div>
    <div class="code-callout__body">
      <p>For each k from 1 to <code>max_k</code>, a fresh KNN is evaluated with 5-fold CV; mean scores are collected and plotted to show the accuracy curve.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="26-30" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Return Best k</span>
    </div>
    <div class="code-callout__body">
      <p><code>argmax</code> on the scores list finds the index of the highest CV accuracy; the corresponding k value is printed and returned.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/4-advanced_fig_2.png" alt="4-advanced" />
<figcaption>Figure 2: Finding the Best k Value</figcaption>
</figure>

```
Best k value: 6
```

## 4. Handling Imbalanced Data

When one class is much more common than others, KNN can be biased. Here's how to fix it:

#### SMOTE + KNN pipeline with cross-validated accuracy

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

def handle_imbalanced_data(X, y):
    """Balance classes using SMOTE"""
    # Create pipeline with SMOTE and KNN
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])

    # Train and evaluate
    scores = cross_val_score(pipeline, X, y, cv=5)
    print(f"Average accuracy after balancing: {scores.mean():.3f}")

    return pipeline

# Example: Credit Card Fraud Detection
# (Typically has very few fraud cases)
X_imbalanced = np.array([...])  # Your features
y_imbalanced = np.array([...])  # Your labels (mostly 0s, few 1s)
model = handle_imbalanced_data(X_imbalanced, y_imbalanced)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-11" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">SMOTE Pipeline</span>
    </div>
    <div class="code-callout__body">
      <p><code>SMOTE</code> generates synthetic minority-class samples; wrapping it in <code>imblearn.Pipeline</code> with the KNN classifier ensures oversampling only happens inside each CV fold, preventing data leakage.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-23" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Evaluate and Return</span>
    </div>
    <div class="code-callout__body">
      <p><code>cross_val_score</code> evaluates the full pipeline on held-out folds; the placeholder arrays illustrate the expected shape — swap with your real imbalanced dataset.</p>
    </div>
  </div>
</aside>
</div>

## 5. Optimizing for Speed: Using Tree Structures

For large datasets, KNN can be slow. Tree structures help speed it up:

#### `BallTree`-backed k-NN with majority vote via `bincount`

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
from sklearn.neighbors import BallTree

class FastKNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        """Build a ball tree for faster searches"""
        self.tree = BallTree(X)
        self.y_train = y

    def predict(self, X):
        """Make predictions using the ball tree"""
        # Find k nearest neighbors quickly
        distances, indices = self.tree.query(X, k=self.k)

        # Get predictions
        predictions = []
        for idx_set in indices:
            k_labels = self.y_train[idx_set]
            prediction = np.bincount(k_labels).argmax()
            predictions.append(prediction)

        return np.array(predictions)

# Example usage
fast_knn = FastKNN(k=5)
fast_knn.fit(X_train, y_train)
predictions = fast_knn.predict(X_test)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-2" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports</span>
    </div>
    <div class="code-callout__body">
      <p><code>BallTree</code> is a spatial index that finds nearest neighbors in O(log n) rather than the O(n) brute-force scan of default KNN.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="4-12" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Fit: Build the Tree</span>
    </div>
    <div class="code-callout__body">
      <p><code>BallTree(X)</code> builds the index once at fit time; the training labels are stored separately for the voting step during prediction.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="14-24" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Predict: Majority Vote</span>
    </div>
    <div class="code-callout__body">
      <p><code>tree.query</code> returns the k nearest neighbor indices per query point; <code>np.bincount(...).argmax()</code> picks the most frequent class label among them.</p>
    </div>
  </div>
</aside>
</div>

## Common Mistakes to Avoid

1. **Using Weighted KNN Without Scaling**

   #### Scale features when using `weights='distance'`

   ```python
   #  Wrong way
   knn = KNeighborsClassifier(weights='distance')
   knn.fit(X_unscaled, y)
   
   #  Right way
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   knn.fit(X_scaled, y)
   ```

2. **Reducing Dimensions Too Much**

   #### Fixed 1-D PCA vs variance threshold

   ```python
   #  Wrong way
   pca = PCA(n_components=1)  # Too few components
   
   #  Right way
   pca = PCA(n_components=0.95)  # Keep 95% of variance
   ```

3. **Ignoring Class Imbalance**

   #### Train on raw imbalance vs SMOTE-resampled data

   ```python
   #  Wrong way
   knn = KNeighborsClassifier()
   knn.fit(X_imbalanced, y_imbalanced)
   
   #  Right way
   from imblearn.over_sampling import SMOTE
   smote = SMOTE()
   X_balanced, y_balanced = smote.fit_resample(X_imbalanced, y_imbalanced)
   knn.fit(X_balanced, y_balanced)
   ```

## Best Practices

1. **Always Scale Your Data**

   #### `StandardScaler` on `X`

   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **Use Cross-Validation**

   #### Mean 5-fold accuracy for `knn` on scaled data

   ```python
   from sklearn.model_selection import cross_val_score
   scores = cross_val_score(knn, X_scaled, y, cv=5)
   print(f"Average accuracy: {scores.mean():.3f}")
   ```

3. **Try Different Distance Metrics**

   #### Compare `metric` strings with the same CV setup

   ```python
   metrics = ['euclidean', 'manhattan', 'cosine']
   for metric in metrics:
       knn = KNeighborsClassifier(metric=metric)
       score = cross_val_score(knn, X_scaled, y, cv=5).mean()
       print(f"{metric}: {score:.3f}")
   ```

## Gotchas

- **Using `weights='distance'` without scaling first** — Distance-based weighting amplifies the dominance of unscaled features: if one feature has values in the thousands, it will virtually eliminate the contribution of all other features to the weight calculation. Always scale before enabling distance weighting.
- **Applying SMOTE outside the cross-validation loop** — The `imblearn.pipeline.Pipeline` example is deliberately correct, but a common mistake is to `SMOTE.fit_resample(X, y)` on the entire dataset before splitting into folds. This leaks synthetic minority-class information into validation folds, inflating reported CV accuracy on imbalanced problems.
- **Reducing dimensions before splitting train and test** — Fitting PCA on `X` (combined train+test) before splitting leaks test distribution information into the PCA components. Always `pca.fit_transform(X_train)` and `pca.transform(X_test)` using the same fitted object.
- **Selecting the best k from a CV curve then re-evaluating on the same test set** — The `find_best_k` function uses CV (correct), but if you then also report accuracy on a held-out `X_test` that was used to verify the chosen k, the test estimate is optimistically biased. Reserve the test set strictly for final reporting.
- **Using `np.bincount` in `FastKNN` when labels are not consecutive integers** — `np.bincount` requires non-negative integer labels starting from 0. If your class labels are arbitrary integers (e.g., `[-1, 1]` or `[2, 5, 10]`), `bincount` either errors or produces wrong argmax results. Use `Counter` or remap labels to 0-based indices first.
- **Ignoring that BallTree fails on high-dimensional data** — `BallTree` and `KDTree` lose their speed advantage over brute force once dimensionality exceeds roughly 20. In high dimensions, the tree degenerates and query time approaches O(n), so the `FastKNN` class provides no benefit without prior dimensionality reduction.

## Additional Resources

For more learning:

- [Scikit-learn KNN Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [PCA Visualization Guide](https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html)
- [Handling Imbalanced Data](https://imbalanced-learn.org/stable/)

Remember: Advanced techniques are tools in your toolbox. Use them when they make sense for your specific problem!
