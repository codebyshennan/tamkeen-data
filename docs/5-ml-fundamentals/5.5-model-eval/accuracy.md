---
reading_minutes: 12
objectives:
  - >-
    Define accuracy as the fraction of correct predictions and compute it for
    binary and multi-class problems with `sklearn.metrics.accuracy_score`.
  - >-
    Recognise when accuracy is misleading: with class imbalance (>~80%
    majority), report at least one of precision/recall, F1, or a confusion
    matrix alongside it.
  - >-
    Compare your model's accuracy to the **majority-class baseline** before
    celebrating, beating "always predict the most common label" is the actual
    bar.
  - >-
    Validate with cross-validation and a held-out test set, not the training
    set, to avoid mistaking memorisation for skill.
---

# Accuracy

**After this lesson:** you can explain Accuracy and try the examples in your own notebook.

## Overview

**Accuracy** as a baseline fraction correct, when it is meaningful and when **class imbalance** makes it misleading.

Relates to [confusion matrix](confusion-matrix.md) and [metrics](metrics.md).

## Introduction

Accuracy is one of the most fundamental metrics in machine learning, measuring the proportion of correct predictions made by a model. While simple to understand and calculate, it's important to use accuracy appropriately and understand its limitations.

## What is Accuracy?

Accuracy is the ratio of correct predictions to total predictions:

\\\[ \text{Accuracy} = \frac{\text{Number of Correct Predictions\}}{\text{Total Number of Predictions\}} \\]

> **Key idea:** accuracy answers **"how often was the model correct overall?"** It does not answer **which class was missed** or **which mistake is expensive**.

**Rule of thumb:** if your dataset has < 80% majority class, accuracy is still OK as a sanity check. Beyond that, always report at least one additional metric.

## Types of Accuracy

### 1. Binary Classification

#### Compute accuracy on a held-out set (binary)

Data and Split

Generate a balanced binary classification dataset and split 80/20; the class balance here makes accuracy a meaningful baseline metric.

Train and Score

Fit logistic regression, call `predict` for hard labels, then pass both to `accuracy_score`-the fraction of matching indices across all test samples.

```
Accuracy: 0.810
```

### 2. Multi-class Classification

#### Accuracy with three classes (Iris)

Iris Setup

Load the three-class Iris dataset and split into train/test; with only 150 samples, `test_size=0.2` reserves 30 samples for evaluation.

Random Forest Accuracy

Fit a Random Forest and measure accuracy; for well-separated Iris classes this typically reaches 1.0, illustrating that accuracy is reliable when classes are balanced and separable.

```
Accuracy: 1.000
```

## Interpreting Accuracy

### 1. Binary Classification

* Perfect accuracy: 1.0
* Random guessing: 0.5
* Worst case: 0.0

### 2. Multi-class Classification

* Perfect accuracy: 1.0
* Random guessing: 1/n\_classes
* Worst case: 0.0

### 3. Balanced vs. Imbalanced Data

* Balanced: Accuracy is meaningful
* Imbalanced: May be misleading
* Consider other metrics

> **Highlight:** compare accuracy to the **majority-class baseline** before celebrating. If 90% of cases are negative, a 90% accurate model may have learned nothing useful.

## Best Practices

1. **Choose Appropriate Metrics**
   * Check class distribution before trusting accuracy. If 95% of examples are negative, a model can score 95% by predicting negative every time.
   * Pair accuracy with precision, recall, F1, or ROC/PR metrics when mistakes have asymmetric costs.
   * Inspect the confusion matrix because it shows which classes are being confused; accuracy hides whether errors are concentrated in the most important class.
2. **Handle Class Imbalance**
   * Use class weights or balanced accuracy when each class should contribute fairly to evaluation.
   * Consider recall or precision for the minority class because that is often where the business risk sits.
   * Apply resampling only inside the training fold of a pipeline; resampling before splitting leaks information and inflates performance.
3. **Validate Results**
   * Use cross-validation to see whether accuracy is stable across different train/test splits.
   * Compare training and validation accuracy to detect overfitting; a large gap means the model is memorising training data.
   * Compare with a simple baseline such as the majority-class classifier. A model that barely beats the baseline may not be useful.
4. **Consider Business Impact**
   * Estimate the cost of false positives and false negatives before selecting a threshold.
   * Align the metric with risk tolerance: a screening model may prefer high recall, while an automated action may require high precision.
   * Tune the decision threshold on validation data rather than accepting the default 0.5 cutoff.

## Common Mistakes to Avoid

1. **Relying Solely on Accuracy**
   * Ignoring class imbalance
   * Missing important patterns
   * Overlooking costs
2. **Poor Data Preparation**
   * Not handling missing values
   * Ignoring outliers
   * Skipping preprocessing
3. **Incorrect Interpretation**
   * Not considering baseline
   * Ignoring business context
   * Overlooking costs

## Practical Example: Credit Risk Prediction

Analyze accuracy for a credit risk prediction model:

#### Pipeline accuracy vs majority baseline

Credit Dataset

Generate five financial features and derive a binary approval label from a threshold on credit score, income, and age, the same synthetic credit setup reused across 5.5 examples.

Pipeline and Prediction

A scaler+forest pipeline prevents data leakage; `predict` returns hard labels used to compute the test-set accuracy score.

Accuracy vs Baseline

Report both model accuracy and the majority-class baseline (`max(P(y=1), P(y=0))`); the gap between the two shows how much the model actually learns beyond trivial prediction.

```
Accuracy: 0.970
Baseline Accuracy: 0.555
```

## Gotchas

* **High accuracy on an imbalanced dataset feels like success**: A dataset with 95% negative samples lets a classifier that always predicts "negative" achieve 95% accuracy; this is the accuracy paradox, and the model has learned nothing; always compare model accuracy to the majority-class baseline before declaring success.
* **`model.score(X_test, y_test)` returns accuracy by default for classifiers**: This is easy to overlook when you later switch to a regression problem where `.score` returns R², not MSE; explicitly call `accuracy_score` or the relevant metric function to make your intent clear and avoid silent metric changes.
* **Balanced accuracy is not the same as accuracy on balanced data**: `sklearn.metrics.balanced_accuracy_score` averages recall per class and is appropriate for imbalanced data; it will differ from standard accuracy even on a balanced dataset if per-class recalls are unequal; choose the right function deliberately.
* **Comparing accuracy across different test set sizes**: Accuracy from a 50-sample test is far noisier than accuracy from a 5000-sample test; a 2% gap between two models may be statistically insignificant on 50 samples; always report confidence intervals or use statistical tests when comparing models on small test sets.
* **Using accuracy as the scoring metric inside `GridSearchCV` for imbalanced problems**: Setting `scoring='accuracy'` in `GridSearchCV` selects the model that maximises accuracy, which on imbalanced data rewards the model that best predicts the majority class; use `scoring='f1'`, `'roc_auc'`, or a custom scorer aligned with your actual objective.

## Additional Resources

1. Scikit-learn documentation on accuracy metrics
2. Research papers on classification metrics
3. Online tutorials on model evaluation
4. Books on machine learning evaluation
