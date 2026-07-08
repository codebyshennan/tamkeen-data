---
reading_minutes: 20
objectives:
  - >-
    Walk through bootstrap aggregating (bagging) with replacement and explain
    how it reduces variance compared with a single tree.
  - >-
    Compute Gini impurity / entropy at a split, and understand why limiting
    `max_features` decorrelates trees.
  - >-
    Read out-of-bag (OOB) error as a free validation estimate, and rank features
    with Mean Decrease in Impurity vs Permutation Importance.
---

# Mathematical Foundation of Random Forest

**After this lesson:** you can explain Mathematical Foundation of Random Forest and try the examples in your own notebook.

## Overview

Explains **bagging**, random subspaces at each split, and why variance drops when trees are decorrelated.

[Introduction](1-introduction.md); compare with [decision trees in 5.2](../../5.2-supervised-learning-1/decision-trees/1-introduction.md).

## Bootstrap Aggregating (Bagging)

### What is Bagging?

Imagine you're trying to understand how people feel about a new movie. Instead of asking just one person, you:

1. Randomly select people from the audience
2. Some people might be asked multiple times
3. Each group gives you a different perspective

This is exactly how bagging works in Random Forest!

### Mathematical Definition

For a dataset of size n, we create m new datasets by randomly sampling with replacement. Each data point has about a 63.2% chance of being selected in each sample.

Function Signature

Takes feature matrix `X` and target `y`; the docstring documents inputs and outputs, critical since \~36.8% of samples will be left out (OOB) for each tree.

Sample with Replacement

`np.random.choice(..., replace=True)` selects n indices from 0…n-1 allowing duplicates, this is bootstrapping, the core mechanism that makes each tree in the forest see different training data.

### Out-of-Bag (OOB) Estimation

Think of this as a built-in validation set. For each tree, some data points weren't used in training - we can use these to estimate how well the model will perform on new data.

## Random Feature Selection

### What is Feature Selection?

Imagine each expert in our committee only looks at certain aspects of a car:

* One expert might focus on safety features
* Another might look at fuel efficiency
* A third might consider price and maintenance costs

This is how Random Forest selects features - each tree only considers a random subset of features when making decisions.

![Feature Importance](<../../../../.gitbook/assets/feature_importance (3).png>) _Figure 1: Feature importance shows which features contribute most to the model's predictions._

### Feature Sampling

At each split in a tree, we only consider a random subset of features:

* For classification: typically \\(\sqrt{p}\\) features
* For regression: typically \\(p/3\\) features where \\(p\\) is the total number of features.

Parameters

Takes total feature count and how many to keep; the docstring clarifies that the return is feature _indices_, not values, these indices are then used to slice columns of `X` at each tree split.

Random Subset

`replace=False` ensures no feature is picked twice per split; in sklearn this corresponds to `max_features='sqrt'` (classification) or `max_features='log2'` by convention.

## Ensemble Prediction

### Classification

For classification problems, it's like taking a vote among all the experts. The most common prediction wins!

### Regression

For regression problems, it's like taking the average of all expert opinions. This helps balance out individual biases.

![Ensemble Prediction](../../../../.gitbook/assets/ensemble_prediction.png) _Figure 2: How individual tree predictions combine to form the final ensemble prediction._

## Feature Importance

### What is Feature Importance?

Think of this as understanding which factors matter most in making a decision. For example, in predicting house prices:

* Location might be very important
* Number of bedrooms might be somewhat important
* Color of the walls might not matter much

### Gini Importance

The Gini importance measures how much each feature contributes to reducing uncertainty in the predictions.

Function and Docstring

Takes a 1D label array `y` and returns a scalar impurity score between 0 (pure node) and 0.5 (maximally mixed for two classes).

Gini Calculation

`np.unique` with `return_counts=True` tallies class frequencies; dividing by `len(y)` gives class probabilities; `1 - sum(p²)` implements the Gini formula.

## Error Analysis

### Bias-Variance Tradeoff

Think of this as the balance between:

* **Bias**: How far off our predictions are on average
* **Variance**: How much our predictions vary from one tree to another

Random Forests help reduce variance while maintaining bias, making the model more stable.

![Bias-Variance Tradeoff](<../../../../.gitbook/assets/bias_variance (1).png>) _Figure 3: The bias-variance tradeoff in Random Forests - how model complexity affects predictions._

## Convergence Properties

### Law of Large Numbers

As we add more trees to our forest, the predictions become more stable and reliable. This is like how a larger sample size gives us more confidence in our results.

## Optimization Criteria

### Split Quality

When deciding how to split the data at each node, we look for splits that:

1. Create more homogeneous groups
2. Reduce uncertainty in our predictions

Signature and Docstring

Takes three label arrays (parent node, left child, right child); information gain is the reduction in Gini impurity from the parent to the weighted-average child impurity.

Weighted Impurity Reduction

Weights each child's Gini impurity by its fraction of the total samples (`n_l/n`, `n_r/n`); a larger gain means this split creates purer children, the tree picks the feature and threshold that maximizes this value.

## Hyperparameter Effects

### Number of Trees

* More trees = more stable predictions
* But diminishing returns after a certain point
* Think of it like adding more experts to a committee - after a while, adding more doesn't help much

### Max Features

* Fewer features = more diverse trees
* More features = better individual trees
* It's like deciding how many aspects each expert should consider

### Tree Depth

* Deeper trees = more detailed decisions
* Shallower trees = more general decisions
* It's like deciding how many questions each expert can ask

![Decision Tree vs Random Forest](../../../../.gitbook/assets/decision_tree_boundary.png) _Figure 4: A single decision tree (left) makes simple, piecewise linear decisions, while a Random Forest (right) combines multiple trees to create more complex decision boundaries._

## Gotchas

* **The 63.2% rule only holds for large datasets**: each bootstrap sample contains \~63.2% unique observations when n is large; for small datasets (n < 100) this fraction varies significantly, so OOB estimates become unreliable and a proper validation split is still needed.
* **Gini importance double-counts correlated features**: when two features carry the same information, each will steal splits from the other across trees, making both appear less important than they truly are; this is why permutation importance on a held-out set is preferred.
* **`max_features='sqrt'` is the default for classification but `max_features=1.0` is the default for regression**: forgetting that defaults differ between `RandomForestClassifier` and `RandomForestRegressor` leads to silently different variance-reduction behaviour.
* **Information gain in this implementation uses Gini, not entropy**: sklearn's `RandomForestClassifier` uses Gini impurity by default; switching to `criterion='entropy'` changes split decisions and can produce slightly different trees, but neither is universally better.
* **Adding trees beyond convergence wastes memory without reducing bias**: the Law of Large Numbers guarantees variance converges as tree count grows, but bias is fixed by individual tree depth; no amount of additional trees can fix underfitting caused by shallow `max_depth`.
* **Feature importance scores sum to 1.0 but are not probabilities**: the normalised sum-to-one property is an artefact of the calculation, not a probabilistic statement; a feature with importance 0.4 is not "40% responsible" for predictions.

## Next Steps

Now that you understand the mathematics behind Random Forests, move on to [Implementation](3-implementation.md) to see how to put these concepts into practice!
