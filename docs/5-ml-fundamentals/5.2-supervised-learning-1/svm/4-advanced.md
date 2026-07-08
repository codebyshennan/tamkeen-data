---
reading_minutes: 35
objectives:
  - >-
    Reason about SMO and how `C` controls the bias/variance tradeoff at the
    optimisation level.
  - >-
    Define a custom kernel and plug it into `SVC(kernel=...)` for
    domain-specific similarity.
  - >-
    Scale SVM training to larger datasets with `LinearSVC`, chunked training,
    and parallel grid search (`n_jobs=-1`).
  - >-
    Drive feature selection from SVM coefficient magnitudes (linear) or via
    recursive feature elimination.
---

# Advanced SVM Techniques

**After this lesson:** you can explain Advanced SVM Techniques and try the examples in your own notebook.

## Overview

Deeper optimization and modeling notes (e.g. class weights, nu-SVM hooks), use when defaults are not enough.

## Advanced Optimization Techniques

### Sequential Minimal Optimization (SMO)

SMO is like breaking a big problem into smaller, manageable pieces. Here's why it's useful:

1. **Faster Training**
   * Works on small subsets of data at a time
   * More efficient than traditional methods
   * Better for large datasets
2. **Memory Efficient**
   * Doesn't need to store entire dataset
   * Works well with limited memory
   * Good for big data applications

### Regularization Parameter (C)

The C parameter controls the trade-off between having a wide margin and correctly classifying training points:

![C Parameter Comparison](../../../../.gitbook/assets/C_parameter_comparison.png)

_Figure: Effect of C parameter on decision boundary. Left: Low C (more regularization), Middle: Balanced C, Right: High C (less regularization)._

Here's a complete example showing the impact of different C values:

#### Effect of C on RBF SVC (and optional early stopping sketch)

Imports

Standard SVM imports for this section's two demonstrations: a C-value comparison plot and an early-stopping convergence loop.

Noisy dataset

100 scattered points (class 0) plus 20 points forming a diagonal line through the scatter (class 1). The overlapping geometry means a low-C boundary will sacrifice some training accuracy for generalization.

Split and scale

80/20 split with `StandardScaler` fit on training data. The scaled versions are used inside the helper functions below.

C comparison plot

Trains an RBF SVC for each of three C values and plots side-by-side decision regions. Support vectors are circled in red so you can see how boundary tightness changes with C.

Early stopping loop

Increments `max_iter` one step at a time and checks whether training accuracy has changed by less than `tolerance`. When the score stabilizes, training stops and a final model is refit with that iteration count.

**Explanation:**

* This example demonstrates how different C values affect the decision boundary
* A low C value creates a smoother boundary but may misclassify some points
* A high C value tries to correctly classify all training points, which can lead to overfitting
* The early stopping implementation monitors model convergence to avoid unnecessary iterations
* We track the model's score and stop training when changes become smaller than a tolerance threshold

## Advanced Kernel Techniques

### Custom Kernel Implementation

Sometimes you need a special kernel for your specific problem. Here's a complete example with a custom kernel:

#### Hybrid RBF + polynomial kernel via `kernel='precomputed'`

Imports

Adds `pairwise_kernels` to compute the RBF component of the custom hybrid kernel matrix.

Spiraling dataset

Class 1 is a tightly wound trigonometric spiral, a shape where neither pure RBF nor pure polynomial excels, motivating the hybrid approach.

Hybrid kernel class

`hybrid_kernel` computes a weighted sum (70% RBF + 30% polynomial) and returns an (n × n) Gram matrix. `SVC(kernel='precomputed')` accepts this matrix directly instead of raw features.

Fit and predict

Training stores a copy of `X_train` so that at prediction time the kernel can be computed between the new points and all training points, this is required for precomputed kernels.

Three-kernel comparison

Trains RBF, polynomial, and the custom hybrid side-by-side, then plots their decision regions. Accuracy is shown in each subplot title for a quick apples-to-apples comparison.

**Explanation:**

* We implement a custom kernel that combines the strengths of RBF and polynomial kernels
* The hybrid kernel is a weighted sum: 70% RBF + 30% polynomial
* Custom kernels are useful when standard kernels don't capture the unique patterns in your data
* The SVC model with kernel='precomputed' allows us to provide a pre-computed kernel matrix
* We store the training data to compute the kernel between test and training data during prediction
* The visualization shows how different kernels create different decision boundaries

## Advanced Visualization

### Decision Boundary and Support Vectors Visualization

Visualizing decision boundaries helps understand how SVM works:

#### Decision surface, margins, and support vectors on two moons

Imports

Brings in `make_moons`, a crescent-shaped dataset that produces a visually compelling non-linear decision boundary.

Moons dataset + fit

200 samples with mild noise. `gamma=10` is a high value that makes the RBF kernel very local, each support vector exerts influence only over a small area, producing a tight boundary.

Meshgrid predictions

Dense grid of points is scaled and passed through both `predict` (class regions) and `decision_function` (distance from hyperplane). Both grids are reshaped for contour plotting.

Decision regions + margins

`contourf` fills class regions; `contour` at levels −1, 0, +1 draws the margin boundaries (dashed) and the decision boundary (solid).

Support vector overlay

Support vectors are circled in red using `model.support_` indices. The percentage of all training points that are support vectors indicates model complexity, a high percentage can hint at overfitting.

**Explanation:**

* This visualization shows not just the decision boundary but also the margins
* The solid line is the decision boundary (where decision function = 0)
* The dashed lines are the margins (where decision function = ±1)
* Support vectors are highlighted with red circles
* We display the percentage of points that are support vectors, which indicates model complexity
* A high percentage of support vectors can suggest the model is complex and might overfit

## Performance Optimization

### Memory-Efficient Implementation

For large datasets, memory efficiency is important:

#### Chunked scaling sketch and `LinearSVC(dual=False)`

Imports

Uses `LinearSVC` instead of `SVC`, the primal linear formulation that avoids the O(n²) kernel matrix, making it suitable for large datasets.

Large dataset

10,000 samples with 20 features simulates a scenario where loading the kernel matrix (n × n = 100M floats) into RAM would be prohibitive.

Chunked scaling

Iterates over 1,000-sample chunks to demonstrate the chunk processing pattern. In production you would use `StandardScaler.partial_fit` for true incremental statistics without reloading.

LinearSVC (primal)

`dual=False` solves the primal optimization problem, which is faster when samples outnumber features. Reports training accuracy and iteration count to confirm convergence.

Efficient prediction

A lightweight wrapper that applies the fitted scaler before predicting, the same two-step pattern used throughout, but packaged as a reusable function.

**Explanation:**

* This implementation processes data in chunks to reduce memory usage
* For very large datasets, we can scale features incrementally without loading everything at once
* The LinearSVC is used with dual=False which is more memory-efficient when n\_samples > n\_features
* In real applications with truly huge datasets, you'd implement the transform step in chunks too
* This approach can handle datasets too large to fit in memory all at once

### Parallel Processing for Parameter Tuning

Speed up training with parallel processing:

#### Parallel evaluation of SVC hyperparameter tuples

Imports

Adds `joblib.Parallel` and `delayed` for process-level parallelism, plus `itertools.product` to generate every parameter combination.

Dataset and scaling

1,000 samples, large enough that running 32 CV evaluations sequentially would be noticeably slow, motivating the parallel approach.

Parameter grid

Same 32-combination grid as the GridSearchCV example, but the parallelism is now hand-coded with `joblib` to expose timing and give finer control.

Per-combination evaluator

`evaluate_params` is the unit of work dispatched to each worker. It runs 5-fold CV with `n_jobs=1` (parallelism is at the outer level, not inside each fold).

Parallel dispatch

`Parallel(n_jobs=-1)` spawns one worker per CPU core. `delayed` wraps the function so joblib can serialize and schedule it. Total wall-clock time is logged after all results arrive.

Results and final model

Best parameters are identified by argmax over CV scores. The optional plot visualizes the accuracy-vs-time trade-off for every combination, helping choose between fast-but-good and slow-but-best configurations.

**Explanation:**

* This implementation uses Parallel and delayed from joblib to run parameter evaluation in parallel
* Each parameter combination is evaluated independently using cross-validation
* The approach is much faster than sequential parameter search, especially with many combinations
* We keep track of evaluation time to identify which parameter combinations are more computationally expensive
* The visualization helps understand the trade-off between parameter performance and computational cost

## Advanced Feature Engineering

### Feature Selection with SVM

Select the most important features with SVM-based feature selection:

#### L1 `LinearSVC` + `SelectFromModel`

Imports

Adds `SelectFromModel`, scikit-learn's meta-transformer that uses a fitted estimator's feature importances to keep only the most relevant columns.

Breast cancer dataset

569 samples with 30 numeric features. The goal is to demonstrate that a small subset of features can match or beat the accuracy of using all 30.

L1 feature selector

`penalty='l1'` drives many coefficients to exactly zero, a built-in feature selector. `SelectFromModel(prefit=True)` then drops any feature whose absolute coefficient falls below the threshold.

Apply selector

Calls `select_features_with_svm` on the scaled training and test sets. Both are reduced to the same subset of columns so downstream models see consistent feature spaces.

Accuracy comparison + plot

Trains two `LinearSVC` models, one on all 30 features, one on the selected subset, and prints their test accuracies. The bar chart highlights selected features in red so their relative importance is immediately visible.

```
Original dataset shape: (569, 30)
Number of features selected: 4 out of 30
```

**Explanation:**

* We use LinearSVC with L1 regularization to encourage sparsity (many coefficients become zero)
* The SelectFromModel transformer keeps only features with importance above a threshold
* By default, the 'mean' threshold keeps features with importance above the mean importance

## Gotchas

* **Setting `max_iter` in the early-stopping loop to a very low value**: The early-stopping sketch re-creates a new `SVC` with `max_iter=i` on every iteration, which is an expensive workaround. More critically, very small `max_iter` values (e.g., 1-5) will consistently trigger `ConvergenceWarning`, and the returned model's `score` reflects an unconverged fit, making the convergence check unreliable. Use `LinearSVC` with `max_iter` if you need true early-stopping behavior.
* **Using `LinearSVC` with L1 penalty and forgetting that `dual=True` is incompatible**: `LinearSVC(penalty='l1')` requires `dual=False`. Leaving the default `dual=True` raises a `ValueError`. This is a common copy-paste error when switching between L1 and L2 regularization in the feature selection pipeline.
* **Interpreting `SelectFromModel` threshold as a percentile**: The `threshold` parameter accepts absolute coefficient magnitude values or string shortcuts like `'mean'` or `'median'`. It does not accept percentile strings like `'75%'`. Passing a string other than `'mean'`/`'median'` raises a `ValueError` rather than silently selecting a fraction of features.
* **Comparing parallel grid search results with `GridSearchCV` scores directly**: The parallel parameter search in the performance optimization section uses manual `cross_val_score` calls. These results may differ slightly from `GridSearchCV` because of different random state handling, fold stratification, and pre-dispatch ordering. They are not drop-in replacements for evaluating best parameters.
* **Using `NuSVC` and treating `nu` as equivalent to `1/C`**: `NuSVC` takes a `nu` parameter (0, 1] controlling the upper bound on the fraction of margin errors and the lower bound on the fraction of support vectors. It is not simply the inverse of `C`; the two formulations optimize different objective functions and will produce different decision boundaries on the same data.
* **Applying `PCA` for visualization after fitting SVM on the full feature space**: If you train an SVM on 30 features and then project to 2D with PCA for a decision boundary plot, the 2D projection does not correspond to the SVM's actual decision boundary (which lives in 30D). The plot is misleading because the SVM never saw or used the 2D coordinates.
