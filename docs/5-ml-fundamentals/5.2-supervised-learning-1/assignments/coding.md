# Assignment: Supervised Learning with Naive Bayes, kNN, and Decision Trees

> **Submit your work on Skills Union →** <https://skillsu.com/member/assessment>

## Setup

Run the following code to load the dataset and prepare for all tasks:

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load the digits dataset (1797 samples, 64 features, 10 classes)
digits = load_digits()
X, y = digits.data, digits.target

# Hold out a test set — do not touch until Task 4
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples:     {X_test.shape[0]}")
print(f"Features:         {X_train.shape[1]}")
print(f"Classes:          {len(np.unique(y))}")
```

## Tasks

### 1. Naive Bayes Classifier

- Import `GaussianNB` from `sklearn.naive_bayes`.
- Fit it on `X_train` and `y_train`.
- Predict on `X_test` and store the result as `nb_preds`.
- Print `accuracy_score(y_test, nb_preds)` and a `classification_report`.
- In a comment, note whether any digit class has noticeably lower recall and suggest why.

### 2. k-Nearest Neighbors — Vary k

- Import `KNeighborsClassifier` from `sklearn.neighbors`.
- Scale the features with `StandardScaler` fitted on `X_train` only, then transform both `X_train` and `X_test`.
- Train a kNN classifier for each value of `k` in `[1, 3, 5, 7, 11, 15, 21]`.
- Record the test accuracy for each `k` in a list or dict and print the results.
- Identify and print the value of `k` that gives the highest test accuracy.
- Store predictions from the best-k model as `knn_preds`.

### 3. Decision Tree — Feature Importances

- Import `DecisionTreeClassifier` from `sklearn.tree`.
- Fit a `DecisionTreeClassifier(random_state=42)` on the **unscaled** `X_train` and `y_train`.
- Predict on `X_test` and store as `dt_preds`.
- Print the test accuracy.
- Extract `feature_importances_` from the fitted tree and identify the top 5 most important pixel features (by index).
- Print those indices and their importance scores.

### 4. Model Comparison

- Build a summary table (dict, DataFrame, or printed lines) showing the test accuracy of all three models: Naive Bayes, best-k kNN, and Decision Tree.
- Print the name and accuracy of the best-performing model.
- In 2–3 comment lines, note one reason a particular model might outperform the others on image/pixel data.

## Deliverable

Submit a single Python script that:

1. Runs all four tasks in sequence with no external files required.
2. Prints clearly labelled results for each task (accuracy scores, classification reports, k-vs-accuracy list, top feature importances, final comparison).
3. Includes brief comments explaining any non-obvious choices (e.g., why you scale for kNN but not for the decision tree).

## Hints

<details>
<summary>Show hints</summary>

### 1. Naive Bayes
- **Where:** [Naive Bayes — Introduction](../naive-bayes/1-introduction.md) — "The Two Phases" section; [Naive Bayes — Implementation](../naive-bayes/4-implementation.md) — "Project 2: Medical Diagnosis System" (Gaussian NB on numeric features).
- **Think:** The digits dataset contains pixel intensities — continuous numeric values — so `GaussianNB` is the right variant. No scaling is strictly required for NB, but confirm the feature range. Check the lesson's note on which NB variant expects which input type.

### 2. kNN — Vary k
- **Where:** [kNN — Introduction](../knn/1-introduction.md) — "Impact of k"; [kNN — Implementation](../knn/3-implementation.md) — "Understanding k in KNN".
- **Think:** kNN uses Euclidean distance, so features at very different scales will dominate. Always scale before fitting. The lesson mentions a `√n` rule as a starting point — with ~1400 training samples, that's about k ≈ 37, but the range given here deliberately includes smaller values to observe the bias–variance tradeoff. Collect accuracies in a loop, not by re-running the cell.

### 3. Decision Tree
- **Where:** [Decision Trees — Introduction](../decision-trees/1-introduction.md) — "Key Components"; [Decision Trees — Implementation](../decision-trees/3-implementation.md) — the `feature_importances_` walkthrough.
- **Think:** Decision trees split on individual features, so scaling is not needed. Access `clf.feature_importances_` after fitting; use `np.argsort` in descending order to find the top-5 indices. A pixel importance of 0 means the tree never split on that pixel.

### 4. Model Comparison
- **Where:** [kNN — Implementation](../knn/3-implementation.md); [Naive Bayes — Implementation](../naive-bayes/4-implementation.md); [Decision Trees — Implementation](../decision-trees/3-implementation.md).
- **Think:** Use the same `X_test` (scaled version for kNN, raw for NB and DT) to ensure a fair comparison. Summarise with a simple dict: `{'Naive Bayes': nb_acc, 'kNN (k=N)': knn_acc, 'Decision Tree': dt_acc}`.

### Common pitfalls
- Fitting `StandardScaler` on `X_test` or on all data before splitting will leak test statistics into training — always `fit` only on `X_train`.
- Using `KNeighborsClassifier` without scaling typically produces noticeably lower accuracy on pixel data because different pixel positions can have very different intensity ranges.
- `feature_importances_` sums to 1.0; a top-5 list with very low individual scores (< 0.02 each) suggests the tree is very deep and spreading importance widely — try adding `max_depth=10` to see if the importances concentrate.
- Do not call `model.predict` before `model.fit` — the kNN loop should call `fit` inside each iteration.

</details>
