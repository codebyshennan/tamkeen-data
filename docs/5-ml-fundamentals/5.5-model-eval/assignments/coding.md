# Assignment: Model Evaluation

> **Submit your work on Skills Union →** <https://skillsu.com/member/assessment>

## Setup

Run the following code to prepare a classifier and dataset for all tasks:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load data
data = load_breast_cancer()
X, y = data.data, data.target   # y: 1 = malignant, 0 = benign

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Baseline model — used in Tasks 1, 3, and 4
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_s, y_train)
y_pred = clf.predict(X_test_s)
y_prob = clf.predict_proba(X_test_s)[:, 1]   # probability of class 1

print("Setup complete.")
print(f"Test set size: {len(y_test)} samples, {y_test.sum()} malignant, {(y_test==0).sum()} benign")
```

```
Setup complete.
Test set size: 114 samples, 72 malignant, 42 benign
```

## Tasks

### 1. Confusion Matrix and Derived Metrics

- Import `confusion_matrix` and `ConfusionMatrixDisplay` from `sklearn.metrics`.
- Compute the confusion matrix from `y_test` and `y_pred`.
- Plot it with `ConfusionMatrixDisplay` (display labels: `["benign", "malignant"]`).
- From the confusion matrix values (TP, TN, FP, FN), manually compute and print:
  - Accuracy  = (TP + TN) / (TP + TN + FP + FN)
  - Precision = TP / (TP + FP)
  - Recall    = TP / (TP + FN)
  - F1-score  = 2 × Precision × Recall / (Precision + Recall)
- Verify your manual values by also printing `classification_report(y_test, y_pred)`.

### 2. k-Fold Cross-Validation

- Import `cross_validate` from `sklearn.model_selection`.
- Run 5-fold cross-validation on a fresh `RandomForestClassifier(n_estimators=100, random_state=42)` using the full `X` and `y` (not the train/test split).
  Use `scoring=['accuracy', 'f1']` and `cv=5`.
- Print the per-fold accuracy and F1 scores.
- Print the mean and standard deviation for both metrics.
- In a comment, note whether the standard deviation suggests the model is stable across folds.

### 3. ROC Curve and AUC

- Import `roc_curve` and `roc_auc_score` from `sklearn.metrics`.
- Use `y_prob` (from Setup) to compute the ROC curve: `fpr, tpr, thresholds = roc_curve(y_test, y_prob)`.
- Compute `auc_score = roc_auc_score(y_test, y_prob)`.
- Plot the ROC curve (FPR on x-axis, TPR on y-axis) with:
  - A dashed diagonal line representing a random classifier.
  - The AUC value in the legend: `f"Random Forest (AUC = {auc_score:.3f})"`.
  - Labelled axes and a title "ROC Curve".
- Print the AUC score.

### 4. GridSearchCV for Hyperparameter Tuning

- Import `GridSearchCV` from `sklearn.model_selection`.
- Define a parameter grid for `RandomForestClassifier`:
  ```python
  param_grid = {
      'n_estimators': [50, 100, 200],
      'max_depth':    [None, 10, 20],
  }
  ```
- Wrap the estimator and grid in a `GridSearchCV` with `cv=5`, `scoring='f1'`, and `n_jobs=-1`.
  Fit it on `X_train_s` and `y_train`.
- Print `grid_search.best_params_` and `grid_search.best_score_`.
- Evaluate the best estimator on `X_test_s` and `y_test` and print its final test F1 score.
- In a comment, explain why the CV best score and the final test score may differ.

## Deliverable

Submit a single Python script that:

1. Runs all four tasks in order — no external files needed.
2. Produces the three required plots (confusion matrix display, ROC curve; the CV bar chart is optional).
3. Prints clearly labelled numeric results for each task.
4. Includes comments comparing manual metric calculations to `classification_report` outputs and explaining the GridSearchCV workflow.

## Hints

<details>
<summary>Show hints</summary>

### 1. Confusion Matrix
- **Where:** [Confusion Matrix](../confusion-matrix.md) — "What is a Confusion Matrix? → TP/TN/FP/FN"; [Metrics](../metrics.md) — "Classification Metrics Comparison Table".
- **Think:** `confusion_matrix` returns a 2×2 array for binary classification. Unpack it as `[[TN, FP], [FN, TP]] = cm.ravel()` if you want named variables. Then plug into the formulas exactly as written in the lesson's table. The `classification_report` output should match your manual values (small floating-point differences are expected).

### 2. k-Fold Cross-Validation
- **Where:** [Cross-Validation](../cross-validation.md) — "Why Cross-Validation Matters"; "k-fold" variants.
- **Think:** `cross_validate` (not `cross_val_score`) returns a dict with keys like `'test_accuracy'` and `'test_f1'`. Use `np.mean(...)` and `np.std(...)` to summarise each array. Fitting on all of `X` and `y` (not the train split) gives the most stable CV estimate over the full dataset.

### 3. ROC Curve and AUC
- **Where:** [Metrics](../metrics.md) — "ROC-AUC" row in the comparison table.
- **Think:** `roc_curve` needs the **probability** scores for the positive class (`y_prob`), not the binary predictions. Plot `fpr` on the x-axis and `tpr` on the y-axis. The diagonal from (0,0) to (1,1) represents random guessing (AUC = 0.5) — add it with `plt.plot([0,1],[0,1],'--')`.

### 4. GridSearchCV
- **Where:** [Hyperparameter Tuning](../hyperparameter-tuning.md) — "GridSearchCV"; "Always wrap preprocessing + model in a Pipeline".
- **Think:** `grid_search.best_score_` is the mean CV F1 on the training folds — it is slightly optimistic because the grid was tuned to maximise it. The test F1 (`f1_score(y_test, best_clf.predict(X_test_s))`) is the unbiased estimate. In this task the preprocessing is done outside the search, but in production you would wrap `(scaler, clf)` in a `Pipeline` so the scaler is refitted per fold.

### Common pitfalls
- Passing `y_pred` (binary labels) instead of `y_prob` (probabilities) to `roc_curve` or `roc_auc_score` — you will get a degenerate two-point curve. Always use `predict_proba(...)[: , 1]`.
- Forgetting `stratify=y` in `train_test_split` on imbalanced data — without it, a random split can put most of one class in training and very few in test, making recall and F1 meaningless.
- Reporting `grid_search.best_score_` as the "final test score" — the lesson explicitly warns against this; always evaluate `best_estimator_` on the held-out `X_test_s`.
- Unravelling the confusion matrix in the wrong order: `sklearn` uses the convention `cm[true_class, predicted_class]`, so for binary labels `cm[0,0]=TN, cm[0,1]=FP, cm[1,0]=FN, cm[1,1]=TP`.

</details>
