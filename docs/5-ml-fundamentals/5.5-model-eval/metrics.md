---
reading_minutes: 30
objectives:
  - "Pick the right **classification** metric for the question: accuracy for balanced screening, precision/recall/F1 when classes are imbalanced or asymmetric, ROC-AUC for ranking, log loss for calibrated probabilities."
  - "Pick the right **regression** metric for the question: MAE for outlier robustness, RMSE when large errors matter more, R² for variance explained, MAPE only when targets are far from zero."
  - "Compute every metric here via `sklearn.metrics` and read its signature (`average=`, `greater_is_better`) without surprises."
  - "Always report a baseline (majority class, mean predictor) alongside the model's metric, a number on its own is not interpretable."
---

# Model Evaluation Metrics

**After this lesson:** you can explain Model Evaluation Metrics and try the examples in your own notebook.

## Overview

**Metrics** turn predictions and labels into comparable numbers: accuracy, error rates, calibration, ranking scores, and more. This hub orients you before the dedicated pages on [confusion matrix](confusion-matrix.md), [precision/recall](precision-recall.md), [ROC/AUC](roc-and-auc.md), and [accuracy](accuracy.md). **Prerequisites:** supervised setup from [5.1](../5.1-intro-to-ml/what-is-ml.md); class imbalance intuition helps when accuracy misleads.

## What are Evaluation Metrics?

Think of evaluation metrics as the "scorecard" or "report card" for your machine learning model. Just like how a teacher uses different tests and assignments to evaluate a student's performance, we use different metrics to evaluate how well our model is performing.

> **Key idea:** choose the metric before tuning. Picking the metric after seeing results makes evaluation drift toward the number that looks best.

### Video Tutorial: Model Evaluation Metrics

<div class="video-embed">
<iframe width="560" height="315" src="https://www.youtube.com/embed/Kdsp6soqA7o" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

*StatQuest: Machine Learning Fundamentals: The Confusion Matrix by Josh Starmer*

### Why Metrics Matter

Imagine you're a doctor diagnosing patients. You wouldn't just look at one symptom - you'd consider multiple factors like temperature, blood pressure, and lab results. Similarly, in machine learning, we need multiple metrics to get a complete picture of our model's performance.

## Real-World Analogies

### The Sports Analogy

Think of model evaluation like evaluating a sports team:

- Accuracy is like the win-loss record
- Precision is like the percentage of shots that hit the target
- Recall is like the percentage of opportunities that were converted
- F1-score is like the overall team performance rating

### The Weather Forecast Analogy

Model evaluation is like weather forecasting:

- Accuracy is like how often the forecast is correct
- Precision is like how specific the forecast is
- Recall is like how well we catch all the important weather events
- ROC curve is like the trade-off between false alarms and missed events

## Metrics Comparison and Selection Guide

{% include model-eval-html-diagram.html diagram="metrics" title="Metric selection diagram" %}

*Always report more than one metric, a model with high accuracy on an imbalanced dataset can still be nearly useless.*

> **Highlight:** a metric is only meaningful against a **baseline** and a **decision context**. Report what a trivial model would score before interpreting a trained model.

### Classification Metrics Comparison Table

| Metric | Formula | Range | Best Value | Use When | Pros | Cons |
|--------|---------|-------|------------|----------|------|------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | 0-1 | 1.0 | Balanced classes | Simple, intuitive | Misleading with imbalanced data |
| **Precision** | TP/(TP+FP) | 0-1 | 1.0 | Cost of false positives is high | Focuses on positive prediction quality | Ignores false negatives |
| **Recall** | TP/(TP+FN) | 0-1 | 1.0 | Cost of false negatives is high | Focuses on finding all positives | Ignores false positives |
| **F1-Score** | 2×(Precision×Recall)/(Precision+Recall) | 0-1 | 1.0 | Need balance between precision/recall | Balances precision and recall | May not reflect business priorities |
| **ROC-AUC** | Area under ROC curve | 0-1 | 1.0 | Balanced classes, probability ranking | Threshold-independent | Misleading with imbalanced data |
| **PR-AUC** | Area under PR curve | 0-1 | 1.0 | Imbalanced classes | Good for imbalanced data | Less intuitive than ROC-AUC |

### Regression Metrics Comparison Table

| Metric | Formula | Range | Best Value | Use When | Pros | Cons |
|--------|---------|-------|------------|----------|------|------|
| **MSE** | Σ(y_true - y_pred)²/n | 0-∞ | 0 | Penalize large errors heavily | Differentiable, common | Sensitive to outliers |
| **RMSE** | √(MSE) | 0-∞ | 0 | Same units as target | Interpretable units | Sensitive to outliers |
| **MAE** | Σ\|y_true - y_pred\|/n | 0-∞ | 0 | reliable to outliers | Less sensitive to outliers | Less differentiable |
| **R²** | 1 - SS_res/SS_tot | -∞-1 | 1.0 | Measure explained variance | Normalized, interpretable | Can be negative |
| **MAPE** | (1/n)·Σ(\|y_true − y_pred\| / \|y_true\|) | 0-∞ | 0 | Percentage errors matter | Scale-independent | Undefined for zero values |

### Metric Relationships and Trade-offs

![Model Comparison](assets/model_comparison.png)

#### Precision vs Recall Trade-off

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
# Demonstration of precision-recall trade-off
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def demonstrate_precision_recall_tradeoff():
    # Simulate different threshold scenarios
    thresholds = np.linspace(0.1, 0.9, 9)
    scenarios = {
        'Conservative Model': {'precision': [0.95, 0.92, 0.88, 0.82, 0.75, 0.68, 0.60, 0.52, 0.45],
                              'recall': [0.20, 0.35, 0.48, 0.62, 0.73, 0.81, 0.87, 0.91, 0.94]},
        'Aggressive Model': {'precision': [0.60, 0.58, 0.55, 0.52, 0.48, 0.44, 0.40, 0.36, 0.32],
                            'recall': [0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]},
        'Balanced Model': {'precision': [0.80, 0.78, 0.75, 0.72, 0.68, 0.64, 0.60, 0.55, 0.50],
                          'recall': [0.50, 0.58, 0.65, 0.71, 0.76, 0.80, 0.84, 0.87, 0.90]}
    }

    plt.figure(figsize=(12, 8))

    for model_name, metrics in scenarios.items():
        plt.plot(metrics['recall'], metrics['precision'], 'o-',
                label=f'{model_name}', linewidth=2, markersize=6)

    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision vs Recall Trade-off for Different Model Types', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Add annotations
    plt.annotate('High Precision\nLow Recall\n(Conservative)',
                xy=(0.3, 0.9), fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    plt.annotate('Low Precision\nHigh Recall\n(Aggressive)',
                xy=(0.9, 0.4), fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    plt.annotate('Balanced\nPrecision & Recall',
                xy=(0.7, 0.7), fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

    plt.tight_layout()
    plt.show()

demonstrate_precision_recall_tradeoff()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-17" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Scenario Data</span>
    </div>
    <div class="code-callout__body">
      <p>Hard-code three fictional precision/recall pairs at nine threshold values to represent conservative, aggressive, and balanced model behaviors, no real fit needed for this illustrative diagram.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="19-29" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Overlay Three Curves</span>
    </div>
    <div class="code-callout__body">
      <p>Loop over the scenarios dict to draw each model's recall-vs-precision path; both axes are bounded 0-1 so relative positions are immediately comparable.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="31-45" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Region Annotations</span>
    </div>
    <div class="code-callout__body">
      <p>Three <code>annotate</code> boxes label the conservative (top-left), aggressive (bottom-right), and balanced (center) operating regions to make the trade-off intuitive for readers.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/metrics_fig_1.png" alt="metrics" />
<figcaption>Figure 1: Precision vs Recall Trade-off for Different Model Types</figcaption>
</figure>

#### When to Use Which Metric: Decision Tree

```
Start Here: What type of problem?
│
├── Classification
│   │
│   ├── Are classes balanced?
│   │   ├── Yes → Use Accuracy, ROC-AUC
│   │   └── No → Use Precision, Recall, F1, PR-AUC
│   │
│   ├── What's the cost of errors?
│   │   ├── False Positives costly → Optimize Precision
│   │   ├── False Negatives costly → Optimize Recall
│   │   └── Both equally costly → Optimize F1-Score
│   │
│   └── Need probability ranking? → Use ROC-AUC or PR-AUC
│
└── Regression
    │
    ├── Are there outliers?
    │   ├── Yes → Use MAE, Huber Loss
    │   └── No → Use MSE, RMSE
    │
    ├── Need interpretable units? → Use RMSE, MAE
    ├── Need percentage errors? → Use MAPE
    └── Need explained variance? → Use R²
```

## Classification Metrics

### 1. Accuracy

This is like the percentage of correct answers on a test.

#### Accuracy and confusion matrix on synthetic data

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Create sample dataset
X, y = make_classification(n_samples=1000, n_features=20,
                         n_informative=15, n_redundant=5,
                         random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Visualize confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.savefig('assets/confusion_matrix.png')
    plt.show()

plot_confusion_matrix(y_test, y_pred)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-26" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Data, Model, and Accuracy</span>
    </div>
    <div class="code-callout__body">
      <p>Generate a 1000-sample synthetic dataset, fit logistic regression, and compute accuracy; this block establishes <code>y_pred</code> for the confusion matrix helper below.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="28-49" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Confusion Matrix Helper</span>
    </div>
    <div class="code-callout__body">
      <p>Define <code>plot_confusion_matrix</code>: render the 2×2 matrix as an image with <code>imshow</code> and overlay cell counts as text, choosing white or black text based on relative cell magnitude.</p>
    </div>
  </div>
</aside>
</div>

```
Accuracy: 0.825
```

**Output:**
```
Training samples: 800
Test samples: 200
Features: 20
Classes: 2

Accuracy: 0.825

Performance Metrics:
Precision: 0.817
Recall: 0.809
F1-Score: 0.813

Confusion Matrix:
                Predicted
                Neg    Pos
Actual Neg       89     17
       Pos       18     76

Classification Report:
              precision    recall  f1-score   support

           0       0.83      0.84      0.84       106
           1       0.82      0.81      0.81        94

    accuracy                           0.82       200
   macro avg       0.82      0.82      0.82       200
weighted avg       0.82      0.82      0.82       200
```

### 2. Precision and Recall

These are like the balance between being thorough and being accurate.

#### Scalar P/R/F1 and precision-recall curve

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

# Visualize trade-off
def plot_precision_recall_tradeoff(y_true, y_pred_proba):
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig('assets/precision_recall_curve.png')
    plt.show()

# Get probability predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
plot_precision_recall_tradeoff(y_test, y_pred_proba)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-11" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Scalar P/R/F1</span>
    </div>
    <div class="code-callout__body">
      <p>Compute precision, recall, and F1 from hard labels (<code>y_pred</code>); these single-number summaries collapse the full confusion matrix into interpretable business-facing values.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-29" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">PR Curve Helper</span>
    </div>
    <div class="code-callout__body">
      <p>Define <code>plot_precision_recall_tradeoff</code> using <code>predict_proba[:, 1]</code> scores to sweep the threshold; the resulting curve shows the full operating range beyond a fixed 0.5 cutoff.</p>
    </div>
  </div>
</aside>
</div>

```
Precision: 0.817
Recall: 0.809
F1 Score: 0.813
```

**Output:**
```
Precision-Recall Analysis:
Average Precision Score: 0.823

Precision-Recall Curve Data (first 10 points):
Precision | Recall | Threshold
------------------------------
    1.000 |  0.011 |    0.998
    1.000 |  0.223 |    0.969
    0.955 |  0.223 |    0.965
    0.955 |  0.255 |    0.963
    0.923 |  0.255 |    0.962
    0.923 |  0.574 |    0.827
    0.871 |  0.574 |    0.827
    0.871 |  0.596 |    0.811
    0.848 |  0.596 |    0.781
    0.848 |  0.617 |    0.779

Optimal Threshold Analysis:
- Best F1-Score: 0.813 at threshold 0.502
- Best Precision: 1.000 at threshold 0.998
- Best Recall: 1.000 at threshold 0.000
```

### 3. ROC Curve and AUC

This is like the trade-off between sensitivity and specificity.

#### Plot ROC from predicted probabilities

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('assets/roc_curve.png')
    plt.show()

plot_roc_curve(y_test, y_pred_proba)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-3" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports and Signature</span>
    </div>
    <div class="code-callout__body">
      <p>Import <code>roc_curve</code> and <code>auc</code>; the helper accepts any <code>y_true</code>/<code>y_pred_proba</code> pair, making it reusable across different models in the same notebook.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="4-20" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Compute and Plot ROC</span>
    </div>
    <div class="code-callout__body">
      <p>Sweep thresholds with <code>roc_curve</code>, summarize with <code>auc</code>, then plot FPR vs TPR; the dashed diagonal is the random-classifier baseline (AUC = 0.5).</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/metrics_fig_4.png" alt="metrics" />
<figcaption>Figure 4: Receiver Operating Characteristic (ROC) Curve</figcaption>
</figure>

**Output:**
```
ROC Analysis Results:
AUC Score: 0.914

ROC Curve Data (first 10 points):
False Positive Rate | True Positive Rate | Threshold
--------------------------------------------------
             0.000 |             0.000 |      inf
             0.000 |             0.011 |    0.998
             0.000 |             0.223 |    0.969
             0.009 |             0.223 |    0.965
             0.009 |             0.255 |    0.963
             0.019 |             0.255 |    0.962
             0.019 |             0.574 |    0.827
             0.028 |             0.574 |    0.827
             0.028 |             0.596 |    0.811
             0.057 |             0.596 |    0.781

Performance at Different Thresholds:
- Threshold 0.5: Precision=0.817, Recall=0.809, F1=0.813
- Threshold 0.3: Precision=0.750, Recall=0.915, F1=0.824
- Threshold 0.7: Precision=0.889, Recall=0.681, F1=0.771
```

## Regression Metrics

### 1. Mean Squared Error (MSE)

This is like the average squared difference between predictions and actual values.

#### MSE and prediction scatter for regression

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create regression dataset
X, y = make_regression(n_samples=1000, n_features=20,
                      n_informative=15, noise=0.1,
                      random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.3f}")

# Visualize predictions
def plot_regression_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Regression Predictions vs True Values')
    plt.grid(True)
    plt.savefig('assets/regression_predictions.png')
    plt.show()

plot_regression_predictions(y_test, y_pred)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-23" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Regression Setup and MSE</span>
    </div>
    <div class="code-callout__body">
      <p>Generate a regression dataset with low noise, fit linear regression, and compute MSE; with near-perfect signal the MSE should be close to zero, confirming the model recovers the true coefficients.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="25-40" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Prediction Scatter Plot</span>
    </div>
    <div class="code-callout__body">
      <p>Scatter predicted vs true values; points hugging the red diagonal (y=x) indicate accurate predictions, any systematic deviation reveals model bias.</p>
    </div>
  </div>
</aside>
</div>

```
Mean Squared Error: 0.010
```

**Output:**
```
Dataset Summary:
Training samples: 800
Test samples: 200
Features: 20
Target range: [-699.25, 895.97]
Target mean: -1.67
Target std: 233.65

Model Performance:
Mean Squared Error: 0.010042
Root Mean Squared Error (RMSE): 0.100212
Mean Absolute Error (MAE): 0.077831
R-squared Score: 1.000000

Residual Analysis:
Mean: 0.005114
Std: 0.100081
Min: -0.281454
Max: 0.332576

Model Coefficients (first 10):
Feature | Coefficient
-------------------------
X     0 |   69.725351
X     1 |   18.609466
X     2 |   86.876709
X     3 |   98.574173
X     4 |   -0.000673
X     5 |    0.003768
X     6 |   48.384112
X     7 |    3.102771
X     8 |   79.997339
X     9 |   65.165862

Prediction Examples (first 10 test samples):
True Value | Predicted | Residual
-----------------------------------
  -181.086 |  -180.988 |   -0.098
   -82.667 |   -82.642 |   -0.025
   260.859 |   260.935 |   -0.076
   282.385 |   282.442 |   -0.056
    65.916 |    65.933 |   -0.016
    99.381 |    99.412 |   -0.031
  -221.661 |  -221.752 |    0.091
   206.724 |   206.756 |   -0.032
  -578.084 |  -578.140 |    0.056
   223.480 |   223.213 |    0.268
```

### 2. R-squared Score

This is like the percentage of variance explained by the model.

#### R² and residual plot

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f"R-squared Score: {r2:.3f}")

# Visualize residuals
def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    plt.savefig('assets/residual_plot.png')
    plt.show()

plot_residuals(y_test, y_pred)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-3" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">R² Score</span>
    </div>
    <div class="code-callout__body">
      <p>Compute <code>r2_score</code>, the proportion of variance in <code>y_test</code> explained by the model; 1.0 is perfect fit, 0.0 means the model does no better than predicting the mean.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="5-18" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Residual Plot</span>
    </div>
    <div class="code-callout__body">
      <p>Plot residuals (true − predicted) vs predicted values; residuals scattered randomly around the red horizontal zero line indicate no systematic error or heteroscedasticity.</p>
    </div>
  </div>
</aside>
</div>

```
R-squared Score: 1.000
```

## Common Mistakes to Avoid

1. **Using Wrong Metrics**
   - Using accuracy for imbalanced data
   - Using MSE for classification
   - Not considering business context

2. **Ignoring Data Distribution**
   - Not checking class imbalance
   - Not considering outliers
   - Not validating assumptions

3. **Overlooking Model Limitations**
   - Not considering model bias
   - Not checking for overfitting
   - Not validating on new data

## Practical Example: Credit Risk Prediction

Look at how different metrics help evaluate a credit risk model:

#### Pipeline + multiple classification metrics

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Create credit risk dataset
np.random.seed(42)
n_samples = 1000

# Generate features
age = np.random.normal(35, 10, n_samples)
income = np.random.exponential(50000, n_samples)
credit_score = np.random.normal(700, 100, n_samples)

X = np.column_stack([age, income, credit_score])
y = (credit_score + income/1000 + age > 800).astype(int)  # Binary target

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Calculate metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")

# Plot ROC curve
plot_roc_curve(y_test, y_pred_proba)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-19" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Credit Dataset</span>
    </div>
    <div class="code-callout__body">
      <p>Stack three financial features and derive a binary approval label from a threshold, the same synthetic credit setup used across 5.5 examples for consistent comparisons.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="21-37" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Pipeline and Predictions</span>
    </div>
    <div class="code-callout__body">
      <p>Scale then classify in a single pipeline; extract both hard labels (<code>predict</code>) and probability scores (<code>predict_proba[:, 1]</code>) to feed different metric functions.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="39-46" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Multiple Metrics and ROC</span>
    </div>
    <div class="code-callout__body">
      <p>Print accuracy, precision, recall, and F1 as a summary, then call the reusable <code>plot_roc_curve</code> helper to show the model's full threshold-independent performance.</p>
    </div>
  </div>
</aside>
</div>

```
Accuracy: 0.980
Precision: 0.967
Recall: 0.989
F1 Score: 0.978
```

## Best Practices

1. **Choose Appropriate Metrics**
   - Start from the business objective because the metric should reward the behaviour you actually want.
   - Account for data characteristics such as class imbalance, target scale, outliers, and zero-valued targets before choosing a metric.
   - Use multiple supporting metrics when one number hides an important trade-off, but declare one primary metric before tuning.

2. **Validate Thoroughly**
   - Use cross-validation to estimate how much the metric varies across splits.
   - Check multiple data splits when the dataset is small or noisy; unstable metrics should be reported with uncertainty.
   - Test on holdout data only after model and threshold choices are fixed, otherwise the holdout becomes another validation set.

3. **Monitor Over Time**
   - Track metric stability after deployment because real data distribution can shift away from the training sample.
   - Watch for degradation by segment, not just overall; a model can remain stable on average while failing for one group.
   - Update baselines when the population, product, or labelling process changes so comparisons remain meaningful.

## Gotchas

- **Reporting only accuracy on an imbalanced dataset**: A model that predicts "no fraud" 100% of the time achieves 99% accuracy on a dataset with 1% fraud; the accuracy number looks excellent while the model provides zero value; always report at least precision, recall, or F1 alongside accuracy when classes are unequal.
- **Using MSE to compare models trained on different target scales**: MSE for house prices measured in dollars will dwarf MSE for a model predicting prices in thousands of dollars, even if the models are equally good; always report RMSE in the target's original units, or use R² to make scale-independent comparisons.
- **MAPE silently breaks on zero-valued targets**: `mean_absolute_percentage_error` divides by the true value; if any `y_true` element is 0 the result is infinite or undefined, often silently returning `NaN`; check your target distribution for zeros before choosing MAPE.
- **Choosing a metric after seeing results**: Deciding to switch from accuracy to F1 after noticing that accuracy looks bad is p-hacking for ML; choose your primary metric before training, based on the business problem, and stick to it as the final decision criterion.
- **Passing hard predictions to `roc_auc_score`**: `roc_auc_score` needs probability scores or decision-function values, not binary predictions. Passing `model.predict(X_test)` does **not** error or warn and does **not** return 0.5, it silently computes a wrong-but-plausible value equal to the *balanced accuracy* of the hard predictions, which understates the model's true ranking AUC. In the logistic-regression example on this page, `roc_auc_score(y_test, model.predict(X_test))` returns **0.824** (the balanced accuracy of the 0/1 predictions) with no warning, while the correct `roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])` returns **0.914**. A silent, believable wrong number is more dangerous than an obvious 0.5, so always pass probabilities or scores.
- **Treating R² near 1.0 as proof of a good model**: On a dataset with a strong linear trend, even a bad model can achieve R²=0.95 because the metric measures variance explained relative to the mean, not prediction error in absolute terms; always inspect residual plots alongside R² to detect heteroscedasticity or systematic bias.

## Additional Resources

- [Scikit-learn Metrics Guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Model Evaluation Best Practices](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
- [Classification Metrics Tutorial](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)

Remember: Choose metrics that align with your business goals and data characteristics!
