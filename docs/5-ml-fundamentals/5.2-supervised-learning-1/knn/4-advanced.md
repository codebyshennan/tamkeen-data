---
reading_minutes: 20
objectives:
  - >-
    Switch to `weights='distance'` so closer neighbours count more when class
    densities vary.
  - Apply PCA before kNN when `p` is large to fight the curse of dimensionality.
  - >-
    Tune `n_neighbors` with `GridSearchCV` and stratified k-fold to avoid noisy
    single-split estimates.
  - >-
    Trade exact for fast neighbour search using `algorithm='kd_tree'` or
    `'ball_tree'` on larger datasets.
---

# Advanced KNN Techniques: Taking Your Skills to the Next Level

**After this lesson:** you can explain Advanced KNN Techniques: Taking Your Skills to the Next Level and try the examples in your own notebook.

## Overview

Touches **curse of dimensionality**, approximate neighbors, and practical tuning when $p$ is large or classes are imbalanced.

## Why Advanced Techniques Matter

Advanced KNN techniques help you:

* Handle complex real-world data better
* Improve model accuracy
* Make predictions faster
* Deal with special cases like imbalanced data

## 1. Weighted KNN: Giving More Importance to Closer Neighbors

Think of weighted KNN like asking your friends for movie recommendations:

* Your best friend's opinion matters more than a casual acquaintance
* The closer someone is to you, the more you trust their recommendation

### How Weighted KNN Works

#### `weights='distance'` on synthetic genre features

Weighted KNN Setup

Six movies labeled by genre with three score features; `weights='distance'` makes nearer neighbors cast stronger votes than distant ones, helpful when one cluster is much closer to the query.

Predict Genre

The new movie with balanced genre scores is passed to `predict`; the weighted vote among the 5 nearest neighbors determines the output genre.

```
Predicted genre: Action
```

## 2. Dimensionality Reduction: Making Complex Data Simpler

Sometimes your data has too many features, making KNN slow and less accurate. Dimensionality reduction helps by:

* Reducing the number of features
* Keeping the most important information
* Making visualization easier

### Using PCA (Principal Component Analysis)

#### PCA to 2D and scatter-plot Iris by class

PCA Reduction

`PCA(n_components=2).fit_transform` compresses all four Iris features into two principal components that capture the most variance, enabling a 2D scatter plot of a 4D dataset.

Scatter with Colorbar

Load Iris and call the helper; `c=y` colors each point by class, and the colorbar maps numeric label to color, clear cluster separation indicates the first two PCs are informative.

<figure><img src="../../../../.gitbook/assets/4-advanced_fig_1 (1).png" alt="4-advanced"><figcaption><p>Figure 1: Data Visualization After PCA</p></figcaption></figure>

## 3. Finding the Best k Value

Choosing the right number of neighbors (k) is important. Too few can lead to noise, too many can blur boundaries.

### Cross-Validation for k Selection

#### Sweep `k` with 5-fold CV and plot accuracy vs `k`

Imports

NumPy for argmax, `KNeighborsClassifier` for fitting, `cross_val_score` for unbiased k-selection, and matplotlib for the curve plot.

CV Sweep

For each k from 1 to `max_k`, a fresh KNN is evaluated with 5-fold CV; mean scores are collected and plotted to show the accuracy curve.

Return Best k

`argmax` on the scores list finds the index of the highest CV accuracy; the corresponding k value is printed and returned.

<figure><img src="../../../../.gitbook/assets/4-advanced_fig_2 (1).png" alt="4-advanced"><figcaption><p>Figure 2: Finding the Best k Value</p></figcaption></figure>

```
Best k value: 6
```

## 4. Handling Imbalanced Data

When one class is much more common than others, KNN can be biased. Here's how to fix it:

#### SMOTE + KNN pipeline with cross-validated accuracy

SMOTE Pipeline

`SMOTE` generates synthetic minority-class samples; wrapping it in `imblearn.Pipeline` with the KNN classifier ensures oversampling only happens inside each CV fold, preventing data leakage.

Evaluate and Return

`cross_val_score` evaluates the full pipeline on held-out folds; the placeholder arrays illustrate the expected shape, swap with your real imbalanced dataset.

## 5. Optimizing for Speed: Using Tree Structures

For large datasets, KNN can be slow. Tree structures help speed it up:

#### `BallTree`-backed k-NN with majority vote via `bincount`

Imports

`BallTree` is a spatial index that finds nearest neighbors in O(log n) rather than the O(n) brute-force scan of default KNN.

Fit: Build the Tree

`BallTree(X)` builds the index once at fit time; the training labels are stored separately for the voting step during prediction.

Predict: Majority Vote

`tree.query` returns the k nearest neighbor indices per query point; `np.bincount(...).argmax()` picks the most frequent class label among them.

## Common Mistakes to Avoid

1.  **Using Weighted KNN Without Scaling**

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
2.  **Reducing Dimensions Too Much**

    #### Fixed 1-D PCA vs variance threshold

    ```python
    #  Wrong way
    pca = PCA(n_components=1)  # Too few components

    #  Right way
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    ```
3.  **Ignoring Class Imbalance**

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

1.  **Always Scale Your Data**

    #### `StandardScaler` on `X`

    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ```
2.  **Use Cross-Validation**

    #### Mean 5-fold accuracy for `knn` on scaled data

    ```python
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(knn, X_scaled, y, cv=5)
    print(f"Average accuracy: {scores.mean():.3f}")
    ```
3.  **Try Different Distance Metrics**

    #### Compare `metric` strings with the same CV setup

    ```python
    metrics = ['euclidean', 'manhattan', 'cosine']
    for metric in metrics:
        knn = KNeighborsClassifier(metric=metric)
        score = cross_val_score(knn, X_scaled, y, cv=5).mean()
        print(f"{metric}: {score:.3f}")
    ```

## Gotchas

* **Using `weights='distance'` without scaling first**: Distance-based weighting amplifies the dominance of unscaled features: if one feature has values in the thousands, it will virtually eliminate the contribution of all other features to the weight calculation. Always scale before enabling distance weighting.
* **Applying SMOTE outside the cross-validation loop**: The `imblearn.pipeline.Pipeline` example is deliberately correct, but a common mistake is to `SMOTE.fit_resample(X, y)` on the entire dataset before splitting into folds. This leaks synthetic minority-class information into validation folds, inflating reported CV accuracy on imbalanced problems.
* **Reducing dimensions before splitting train and test**: Fitting PCA on `X` (combined train+test) before splitting leaks test distribution information into the PCA components. Always `pca.fit_transform(X_train)` and `pca.transform(X_test)` using the same fitted object.
* **Selecting the best k from a CV curve then re-evaluating on the same test set**: The `find_best_k` function uses CV (correct), but if you then also report accuracy on a held-out `X_test` that was used to verify the chosen k, the test estimate is optimistically biased. Reserve the test set strictly for final reporting.
* **Using `np.bincount` in `FastKNN` when labels are not consecutive integers**: `np.bincount` requires non-negative integer labels starting from 0. If your class labels are arbitrary integers (e.g., `[-1, 1]` or `[2, 5, 10]`), `bincount` either errors or produces wrong argmax results. Use `Counter` or remap labels to 0-based indices first.
* **Ignoring that BallTree fails on high-dimensional data**: `BallTree` and `KDTree` lose their speed advantage over brute force once dimensionality exceeds roughly 20. In high dimensions, the tree degenerates and query time approaches O(n), so the `FastKNN` class provides no benefit without prior dimensionality reduction.

## Additional Resources

For more learning:

* [Scikit-learn KNN Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
* [PCA Visualization Guide](https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html)
* [Handling Imbalanced Data](https://imbalanced-learn.org/stable/)

Remember: Advanced techniques are tools in your toolbox. Use them when they make sense for your specific problem!
