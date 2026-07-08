---
reading_minutes: 30
objectives:
  - >-
    Use `ExtraTreesClassifier` / `ExtraTreesRegressor` and recognise when extra
    randomization helps over standard random forests.
  - >-
    Handle class imbalance with `class_weight`, `sample_weight`, and resampling
    strategies (e.g., SMOTE) inside a forest pipeline.
  - >-
    Plug a random forest into a serving pipeline, estimate model size, run batch
    inference, and persist with pickle/joblib.
---

# Advanced Random Forest Techniques

**After this lesson:** you can explain Advanced Random Forest Techniques and try the examples in your own notebook.

## Overview

Extra trees, class imbalance handling, and tuning strategies beyond defaults.

## Ensemble Optimization

### 1. Stacking with Random Forests

Three Forest Variants

Three `RandomForestClassifier` instances differ in depth, feature sampling strategy, and leaf size, diversity in the base models is key; similar models won't add information when stacked.

Stack and Fit

`StackingClassifier` generates out-of-fold predictions from each base model with `cv=5`, then trains `LogisticRegression` to combine them, a learned ensemble that outperforms majority voting.

### 2. Weighted Voting

Collect Probabilities

List-comprehension calls `predict_proba` on each model and stacks results into a 3D array (models × samples × classes) for vectorized aggregation.

Weighted Average

`np.average(..., weights=weights, axis=0)` combines probability matrices; `argmax` then picks the class with the highest combined probability as the final prediction.

## Advanced Feature Engineering

### 1. Automated Feature Interactions

Setup

Copy the dataframe to avoid in-place mutation; collect column names for combinatorial generation up to the specified degree.

Product Columns

For each feature combination, initialize a new column to 1 then multiply by each component feature; the starred join creates readable column names like `age*income` for downstream interpretation.

### 2. Feature Selection with Permutation Importance

Permutation Importance

`permutation_importance` shuffles each feature 10 times and measures the drop in model score, features that cause a large drop are important; those that don't can be dropped.

Importance DataFrame

Package mean and std importance into a sorted DataFrame, the std across 10 repeats shows how stable each feature's importance is, helping distinguish truly important features from noisy ones.

## Optimization Techniques

### 1. Dynamic Feature Selection

Class Setup

The selector wraps any `base_model` that exposes `feature_importances_`; the `threshold` controls how aggressively low-importance features are dropped.

Two-Pass Fit

The model is first fit on all features to compute importances; features above the threshold are retained and the model is refit on only those, reducing noise and inference cost.

Predict

Prediction uses only the selected feature subset stored during fit, so test data is automatically filtered to match the training column set.

### 2. Memory-Efficient Implementation

Class Setup

Stores target tree count and an empty list that will hold each individually trained tree; fitting in batches avoids materializing all trees in memory at once.

Batch Training

Trees are built in groups of `batch_size`; each single-estimator RF is fit on a fresh bootstrap sample, then appended to the list, keeping peak memory proportional to one batch.

Majority Vote

All trees predict; `np.apply_along_axis` with `bincount(...).argmax()` picks the most frequent class label across the ensemble for each sample.

## Advanced Evaluation Metrics

### 1. Custom Evaluation Framework

Class Setup

Wraps any fitted RF model to add two post-hoc analysis methods: a stability score across bootstrap refits and bootstrap confidence intervals for each feature's importance.

Stability Score

The model is refit on 10 bootstrap samples; the standard deviation of importances across runs captures how consistently each feature is ranked, converted to a 0-1 stability score.

Confidence Intervals

1 000 bootstrap refits build a distribution of importances per feature; percentile-based lower and upper bounds form a 95% CI returned as a tidy DataFrame for reporting.

## Interpretability Techniques

### 1. Partial Dependence Plots

Subplot Grid

Creates one subplot per feature with 5-inch height each; dynamically scaling the figure height keeps plots readable regardless of how many features are requested.

PDP per Feature

`partial_dependence(kind='average')` returns grid values and average predictions; plotting `pdp[1][0]` vs `pdp[0][0]` shows how the marginal prediction changes across the feature's range.

### 2. SHAP Values

TreeExplainer

`shap.TreeExplainer` uses a tree-path algorithm to compute exact SHAP values efficiently for tree-based models, much faster than the model-agnostic kernel SHAP approach.

Summary Plot

`shap.summary_plot` shows a beeswarm of SHAP values per feature, each point is one sample, color encodes feature value, and x-position shows the impact on model output.

## Production Deployment

### 1. Model Versioning

Class Setup

A `RandomForestClassifier` is created with forwarded kwargs; `version` and `history` will track each training run's metadata for auditability.

Versioned Fit

After training, a snapshot of the timestamp, sample count, and per-feature importances is stored under the current version number; the counter increments for the next call.

Save Version

`joblib.dump` serializes the model, current version number, and full training history together so any saved checkpoint can be fully reconstructed later.

### 2. Online Learning

Class Setup

Two lists act as a rolling buffer for incoming samples; `buffer_size` controls how many new points trigger a full model retrain, balancing freshness against compute cost.

Partial Fit

Each call appends new rows to the buffer; when the buffer reaches `buffer_size`, a fresh `RandomForestClassifier` is trained on all buffered data and the buffer is cleared for the next window.

## Gotchas

* **`StackingClassifier` leaks if base models are trained on the full training set**: sklearn's `StackingClassifier` uses `cv` to generate out-of-fold predictions for the meta-learner by default; manually fitting base models on the whole training set and then stacking them allows the meta-learner to see training predictions, inflating performance estimates.
* **Permutation importance can be misleading when features are correlated**: shuffling one correlated feature still leaves its information accessible through the correlated partner, so both features will look less important than they truly are; prefer partial dependence plots for correlated settings.
* **The `DynamicFeatureSelector` refits on a subset without re-tuning hyperparameters**: after dropping low-importance features and refitting, the original `max_depth` or `n_estimators` may no longer be optimal for the reduced feature set; the two-pass approach needs its own hyperparameter validation.
* **`OnlineRandomForest.partial_fit` loses all historical data on each buffer flush**: the implementation retrains from scratch on the current buffer window only, discarding older examples; this is not true online learning but windowed batch retraining, which can cause catastrophic forgetting on drifting data.
* **SHAP's `TreeExplainer` returns a list of arrays for multi-class classifiers**: `shap_values` is a list of length `n_classes`, not a single 2D array; passing the raw return value to `shap.summary_plot` for a binary classifier will work, but for multi-class you must index into the list (e.g., `shap_values[1]` for class 1).
* **`partial_dependence` with `kind='average'` averages over the marginal distribution of other features**: this can produce unrealistic feature combinations (e.g., a very high income with a very low credit score) that the model was never trained on, leading to extrapolated PDP curves that don't reflect real-world behaviour.

## Next Steps

Ready to see Random Forests in action? Continue to [Applications](5-applications.md) to explore real-world use cases!
