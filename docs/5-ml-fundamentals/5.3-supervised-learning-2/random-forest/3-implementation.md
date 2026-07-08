---
reading_minutes: 30
objectives:
  - >-
    Fit `RandomForestClassifier` / `RandomForestRegressor` end-to-end with
    sensible defaults for `n_estimators`, `max_features`, and
    `min_samples_leaf`.
  - >-
    Read `feature_importances_` and the OOB score, and use permutation
    importance for a fairer comparison when features are correlated.
  - >-
    Tune hyperparameters with `GridSearchCV` / `RandomizedSearchCV` and avoid
    the more-is-always-better trap by tracking validation score, not just
    training fit.
---

# Implementing Random Forest

**After this lesson:** you can explain Implementing Random Forest and try the examples in your own notebook.

## Overview

`RandomForestClassifier` / `Regressor`: `n_estimators`, `max_features`, out-of-bag estimates, and feature importances.

## Basic Implementation

### Simple Classification Example

Start with a basic example that shows how to create and train a Random Forest classifier:

#### `RandomForestClassifier` on synthetic data

Data and Split

1000 samples with 15 informative and 5 redundant features are generated; the 80/20 split with a fixed seed ensures reproducible train/test partitions across notebook reruns.

Fit and Evaluate

100 trees with depth limited to 10 are grown and their votes aggregated; the classification report shows per-class precision, recall, and F1 on the held-out test set.

![Decision Tree vs Random Forest](../../../../.gitbook/assets/decision_tree_boundary.png) _Figure 1: A single decision tree (left) makes simple, piecewise linear decisions, while a Random Forest (right) combines multiple trees to create more complex decision boundaries._

## Real-World Example: Credit Risk Prediction

Create a more realistic example that shows how Random Forest can be used in a real-world scenario:

#### Credit risk: OOB score, probabilities, risk tiers

Feature matrix + risk label

Five financial columns become the feature matrix. The risk label is created with a compound boolean: high debt ratio AND low credit score → 1 (risky). `.astype(int)` converts the boolean to 0/1 integers for the classifier.

Key hyperparameters

`max_features='sqrt'` means each tree sees only √(n\_features) columns per split, the randomness that makes trees diverse. `oob_score=True` uses samples not in each bootstrap bag as a free validation set. `n_jobs=-1` uses all CPU cores.

Out-of-bag score

Because each tree is trained on a bootstrap sample (\~63% of data), the remaining \~37% acts as a held-out test set for that tree. `oob_score_` aggregates these per-tree estimates, a free generalization measure without a separate validation split.

Predict probabilities + risk tiers

`predict_proba` returns probabilities for each class; `[:, 1]` takes the "high risk" column. `pd.cut` bins the continuous risk score into business-friendly categories, more actionable than raw predictions.

```
Out-of-bag score: 0.998

Risk Distribution:
Low       77
Medium     0
High      14
Name: count, dtype: int64
```

## Feature Importance Analysis

Understanding which features are most important for making predictions:

#### Importance bars with tree-to-tree variance

Importance and Variance

`feature_importances_` is the mean impurity decrease over all trees; the standard deviation across individual tree importances quantifies how consistently each feature matters, high std means the importance is unstable.

Sorted Bar Chart

Features are ranked from most to least important; error bars show the per-tree variance, making it easy to distinguish reliably important features from those that vary across trees.

![Feature Importance](<../../../../.gitbook/assets/feature_importance (3).png>) _Figure 2: Feature importance shows which features contribute most to the model's predictions._

## Hyperparameter Tuning

Finding the best combination of parameters for your model:

#### `RandomizedSearchCV` with ROC-AUC

Parameter Space

Six hyperparameters are searched: tree count (100-500 from uniform integer distribution), depth, split thresholds, leaf size, feature subset method, and bootstrap flag, mixing continuous distributions with discrete lists.

Randomised CV Search

100 random configurations are evaluated via 5-fold CV optimising ROC-AUC; `n_jobs=-1` parallelises across all cores, making 100-iter random search practical even with a large parameter space.

![Bias-Variance Tradeoff](<../../../../.gitbook/assets/bias_variance (1).png>) _Figure 3: The bias-variance tradeoff in Random Forests - how model complexity affects predictions._

## Advanced Techniques

### 1. Custom Scorer

Creating a custom scoring metric that favors precision over recall:

#### `make_scorer` + \\(F\_\beta\\) (\\(\beta<1\\) favors precision)

Custom Scorer

`make_scorer` wraps `fbeta_score` so it works inside CV functions; beta=0.5 means precision counts twice as much as recall, useful when false positives are costlier than false negatives.

Cross-validate

5-fold CV returns one score per fold; inspecting the spread reveals whether the model's precision-weighted performance is stable across different data partitions.

```
F-0.5 scores: [0.98214286 1.         1.         1.         0.94339623]
```

### 2. Feature Selection

Selecting only the most important features:

#### `SelectFromModel` thresholded on importances

Threshold Selection

`prefit=True` skips refitting the forest; `threshold='median'` retains only the top half of features by importance, a quick way to halve feature dimensionality.

Transform and Report

`selector.transform` returns a dense array with only the selected columns; `get_support()` provides the boolean mask used to recover feature names from `X.columns`.

```
Selected 3 features
Selected features: ['employment_length', 'debt_ratio', 'credit_score']
```

### 3. Handling Imbalanced Data

Using a balanced version of Random Forest for imbalanced datasets:

#### `BalancedRandomForestClassifier` (imbalanced-learn)

Balanced Bootstrapping

`BalancedRandomForestClassifier` from imbalanced-learn undersamples the majority class in each bootstrap so every tree trains on a balanced subset, reducing the bias toward predicting the majority class.

Fit and Compare

The same train/test split is reused so the classification report can be compared directly with the standard random forest report, typically showing improved minority-class recall at the cost of some precision.

![Ensemble Prediction](../../../../.gitbook/assets/ensemble_prediction.png) _Figure 4: How individual tree predictions combine to form the final ensemble prediction._

## Best Practices

### 1. Model Evaluation

Comprehensive evaluation of model performance:

#### Reports + ROC curve helper

Imports

`RocCurveDisplay.from_estimator` is a convenience method that handles predict\_proba, threshold sweep, and plotting in one call.

Train vs Test Reports

Printing both train and test classification reports in one function makes it easy to spot overfitting: a large gap between train and test precision/recall signals that the model has memorised training examples.

### 2. Feature Engineering

Creating new features to improve model performance:

#### `FunctionTransformer` + `Pipeline`

Feature Engineering

Three derived features are added: two domain-specific ratios (income per age, debt × income) and a squared credit score to capture non-linear effects; copying first avoids mutating the caller's DataFrame.

Pipeline Integration

`FunctionTransformer` wraps the custom function so it participates in sklearn pipelines, the same transformation is automatically applied during both `fit` and `predict`.

### 3. Model Persistence

Saving and loading trained models:

#### `joblib` for sklearn estimators

```python
import joblib

# Save model
joblib.dump(rf, 'random_forest_model.joblib')

# Load model
loaded_rf = joblib.load('random_forest_model.joblib')
```

## Common Pitfalls and Solutions

1.  **Memory Issues**

    * **Purpose:** Cut **RAM** for large `X` and smaller ensembles while prototyping.
    * **Walkthrough:** `float32` halves numeric storage vs `float64`; fewer **`n_estimators`** reduces both memory and the size of `estimators_` in memory.

    ```python
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    # Use smaller data types
    X = X.astype(np.float32)

    # Reduce number of trees
    rf = RandomForestClassifier(n_estimators=50)
    ```
2.  **Long Training Time**

    * **Purpose:** Iterate faster with **shallower** / **fewer-tree** forests before a full run.
    * **Walkthrough:** **`n_jobs=-1`** uses all cores for **`fit`** (and prediction where supported); set on the estimator you actually train.

    ```python
    from sklearn.ensemble import RandomForestClassifier

    # Use fewer trees for initial experiments
    rf_quick = RandomForestClassifier(
        n_estimators=10,
        max_depth=5,
        n_jobs=-1,
    )

    # Or set parallel fits on an existing forest
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    ```

<figure><img src="../../../../.gitbook/assets/3-implementation_fig_1 (3).png" alt="3-implementation"><figcaption><p>Figure 1: Generated visualization</p></figcaption></figure>

3.  **Overfitting**

    * **Purpose:** Constrain **leaf purity** and **depth** so trees do not memorize noise.
    * **Walkthrough:** Larger **`min_samples_leaf`** and lower **`max_depth`** smooth decision boundaries, pair with CV to choose values.

    ```python
    from sklearn.ensemble import RandomForestClassifier

    # Increase min_samples_leaf
    rf = RandomForestClassifier(
        min_samples_leaf=5,
        max_depth=10,
        random_state=42,
    )
    ```

## Gotchas

* **`oob_score=True` requires `bootstrap=True`**: setting `bootstrap=False` and `oob_score=True` together raises a `ValueError`; OOB scoring is only meaningful when bootstrap sampling is active because the out-of-bag samples are the ones not selected by bootstrap.
* **`SelectFromModel` with `prefit=True` freezes the importance threshold at fit time**: if you retrain the forest on new data and call `transform` again without recreating the selector, it still uses the old importance threshold and selected columns, silently producing wrong results.
* **`predict_proba` columns are ordered by `classes_`, not by label value**: on an imbalanced dataset where class 0 happens to be the minority, `predict_proba[:, 1]` is still the probability of class 1 (the one with higher index in `classes_`); always check `rf.classes_` before indexing into the probability matrix.
* **`RandomizedSearchCV` does not set `random_state` on the estimator**: the `random_state=42` in `RandomizedSearchCV` controls which parameter combinations are drawn, not the forest's internal randomness; the `RandomForestClassifier` inside also needs its own `random_state` for reproducible trees.
* **`BalancedRandomForestClassifier` from imbalanced-learn may not be installed**: unlike sklearn estimators, `imblearn` is a separate package; calling it without `pip install imbalanced-learn` raises an `ImportError` with no clear hint about the fix.
* **`joblib.load` on a model saved with a different sklearn version may silently give wrong predictions**: sklearn pickles embed the version; loading across minor versions (e.g., 1.2 → 1.4) usually works but can break if internal estimator structure changed; always record the sklearn version alongside saved models.

## Next Steps

Ready to explore advanced techniques? Continue to [Advanced Topics](4-advanced.md)!
