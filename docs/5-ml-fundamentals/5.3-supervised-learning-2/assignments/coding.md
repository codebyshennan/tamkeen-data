# Assignment: Ensemble Methods and Neural Networks

> **Submit your work on Skills Union →** <https://skillsu.com/member/assessment>

## Setup

Run the following code to prepare the dataset for all tasks:

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load the breast cancer dataset (569 samples, 30 features, 2 classes)
data = load_breast_cancer()
X, y = data.data, data.target

# Stratified split — keep class proportions in each fold
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaled version for algorithms sensitive to feature magnitudes
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")
print(f"Features         : {X_train.shape[1]}")
print(f"Feature names    : {list(data.feature_names[:5])} ...")
print(f"Classes          : {list(data.target_names)}")
```

```
Training samples : 455
Test samples     : 114
Features         : 30
Feature names    : [np.str_('mean radius'), np.str_('mean texture'), np.str_('mean perimeter'), np.str_('mean area'), np.str_('mean smoothness')] ...
Classes          : [np.str_('malignant'), np.str_('benign')]
```

## Tasks

### 1. Random Forest — Feature Importances

- Import `RandomForestClassifier` from `sklearn.ensemble`.
- Fit `RandomForestClassifier(n_estimators=100, random_state=42)` on the **unscaled** `X_train` and `y_train`.
- Predict on `X_test` and store as `rf_preds`. Print accuracy and a `classification_report`.
- Extract `feature_importances_` and pair each value with its feature name from `data.feature_names`.
- Sort by importance (descending) and print the top 5 features with their scores.

### 2. Gradient Boosting — Tune n_estimators

- Import `GradientBoostingClassifier` from `sklearn.ensemble`.
- Train three separate models with `n_estimators` values of `50`, `100`, and `200` (keep `learning_rate=0.1`, `max_depth=3`, `random_state=42` fixed).
- For each, record training accuracy and test accuracy.
- Print all six accuracy values clearly labelled.
- Store predictions from `n_estimators=100` as `gb_preds`.
- In a comment, note any sign of overfitting as `n_estimators` increases.

### 3. MLP Neural Network

- Import `MLPClassifier` from `sklearn.neural_network`.
- Fit `MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)` on the **scaled** `X_train_scaled` and `y_train`.
- Predict on `X_test_scaled` and store as `mlp_preds`. Print accuracy and a `classification_report`.
- Print the number of layers and the total number of learned parameters (`sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_)`).

### 4. Model Comparison

- Collect the test accuracies for Random Forest, Gradient Boosting (n=100), and MLP in a single dict or DataFrame and print a formatted comparison.
- Print the name of the best model.
- In 2–3 comment lines, describe one advantage of ensemble methods over a single decision tree and one advantage of a neural network over a tree-based model.

## Deliverable

Submit a single Python script that:

1. Runs all four tasks in order using only the setup data above (no external files).
2. Prints clearly labelled outputs for each task.
3. Includes brief comments on hyperparameter choices and any observations about overfitting or feature importance.

## Hints

<details>
<summary>Show hints</summary>

### 1. Random Forest
- **Where:** [Random Forest — Introduction](../random-forest/1-introduction.md) — "Key Concepts: Random Feature Selection"; [Random Forest — Implementation](../random-forest/3-implementation.md) — "`feature_importances_`" section.
- **Think:** `rf.feature_importances_` is an array of length 30 (one per feature). Use `np.argsort(importances)[::-1][:5]` to get the top-5 indices, then index into `data.feature_names` for readable names. Random forests do not need scaling — distance metrics are never used.

### 2. Gradient Boosting — Tune n_estimators
- **Where:** [Gradient Boosting — Introduction](../gradient-boosting/1-introduction.md) — "Sequential Learning"; [Gradient Boosting — Implementation](../gradient-boosting/3-implementation.md) — "sklearn GradientBoostingClassifier".
- **Think:** Compare `clf.score(X_train, y_train)` vs `clf.score(X_test, y_test)` at each `n_estimators` value. A rising training score with a flat or falling test score is the overfitting signature. Keep the loop compact — three iterations, same setup, just changing one argument.

### 3. MLP Neural Network
- **Where:** [Neural Networks — Introduction](../neural-networks/1-introduction.md) — "Layers" section; [Neural Networks — Implementation](../neural-networks/3-implementation.md) — "sklearn `MLPClassifier`".
- **Think:** MLP is sensitive to feature scale; always use the scaled data (`X_train_scaled`, `X_test_scaled`). `hidden_layer_sizes=(100, 50)` means two hidden layers with 100 and 50 units. `coefs_` is a list of weight matrices; summing `.size` on each gives total parameters.

### 4. Model Comparison
- **Where:** [Random Forest — Implementation](../random-forest/3-implementation.md); [Gradient Boosting — Implementation](../gradient-boosting/3-implementation.md); [Neural Networks — Implementation](../neural-networks/3-implementation.md).
- **Think:** Use a consistent test set (`X_test` for tree models, `X_test_scaled` for MLP) to keep the comparison fair. A pandas DataFrame with model names as the index and accuracy as a column makes the summary easy to read and sort.

### Common pitfalls
- Forgetting to use `X_test_scaled` (not `X_test`) when predicting with `MLPClassifier` — applying unscaled data to a model trained on scaled data will degrade performance without raising an error.
- Comparing train accuracy instead of test accuracy when picking the best `n_estimators` — a higher train score does not mean better generalisation.
- `feature_importances_` for random forests reflects mean impurity decrease, which can be biased toward high-cardinality features; treat it as a heuristic ranking, not an absolute measure.
- Setting `max_iter` too low for `MLPClassifier` will trigger a `ConvergenceWarning`; increase it or add `early_stopping=True` if you see warnings.

</details>
