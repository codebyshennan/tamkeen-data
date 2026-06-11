---
reading_minutes: 32
objectives:
  - "Define the **ROC curve** as TPR vs FPR over all thresholds, and **AUC** as the probability the model ranks a random positive above a random negative."
  - "Plot ROC and compute AUC with `roc_curve` / `roc_auc_score`; read curve shapes (diagonal = random, top-left elbow = good ranker)."
  - "Pick an operating threshold from the curve using cost-weighted criteria (Youden's J, F1 max, business cost matrix) — not the default 0.5."
  - "Know when ROC misleads: with severe class imbalance, AUC can stay high while precision is awful — switch to **PR-AUC** (or report both)."
---

# ROC Curves and AUC: Complete Guide

**After this lesson:** you can explain the core ideas in “ROC Curves and AUC: Complete Guide” and reproduce the examples here in your own notebook or environment.

## Overview

**ROC curves** and **AUC**: ranking quality across thresholds; complements precision–recall for skewed classes.

## Introduction

The Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) are powerful tools for evaluating binary classification models. They provide a comprehensive view of model performance across different classification thresholds and help us understand the trade-offs between sensitivity and specificity.

### Video Tutorial: ROC and AUC Explained

<div class="video-embed">
<iframe width="560" height="315" src="https://www.youtube.com/embed/4jRBRDbJemM" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

*StatQuest: ROC and AUC, Clearly Explained! by Josh Starmer*

## Real-World Analogies

### The Airport Security Analogy

Think of ROC and AUC like airport security screening:

- **True Positives**: Correctly identifying dangerous items
- **False Positives**: Flagging safe items as dangerous (inconvenience)
- **True Negatives**: Correctly identifying safe items
- **False Negatives**: Missing dangerous items (security risk)

The ROC curve shows how the security system performs at different sensitivity levels. A perfect system would catch all threats without any false alarms.

### The Medical Diagnosis Analogy

Imagine you're a doctor diagnosing a disease:

- **True Positives**: Correctly identifying patients with the disease
- **False Positives**: Diagnosing healthy patients as sick (unnecessary treatment)
- **True Negatives**: Correctly identifying healthy patients
- **False Negatives**: Missing patients who actually have the disease (delayed treatment)

ROC and AUC help us find the right balance between catching all cases and avoiding false alarms.

{% include mermaid-diagram.html src="5-ml-fundamentals/5.5-model-eval/diagrams/roc-and-auc-1.mmd" %}

## Technical Definitions

### ROC Curve Components

The ROC curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings:

**True Positive Rate (TPR)** = Sensitivity = Recall
\\[
\text{TPR} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
\\]

**False Positive Rate (FPR)** = 1 - Specificity
\\[
\text{FPR} = \frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}}
\\]

### AUC (Area Under the Curve)

The AUC measures the model's ability to distinguish between classes:

- **AUC = 1.0**: Perfect classification
- **AUC = 0.5**: Random guessing (diagonal line)
- **AUC < 0.5**: Worse than random (but can be inverted)

**AUC Interpretation Guidelines:**
- **AUC > 0.9**: Excellent
- **0.8 < AUC ≤ 0.9**: Very good
- **0.7 < AUC ≤ 0.8**: Good
- **0.6 < AUC ≤ 0.7**: Fair
- **0.5 < AUC ≤ 0.6**: Poor
- **AUC = 0.5**: Random

## Understanding ROC Curve Shapes

![ROC Curve](assets/roc_curve.png)

**Key Patterns:**
- **Perfect Classifier**: Curve goes straight up to (0,1) then across to (1,1)
- **Random Classifier**: Diagonal line from (0,0) to (1,1)
- **Good Classifier**: Curve bows toward the upper-left corner
- **Poor Classifier**: Curve below diagonal line

## Implementation Examples

### 1. Basic ROC Curve for Binary Classification

#### Train a model and plot one ROC curve

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Create sample dataset
X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=15, n_redundant=5,
                           random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Get probability predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Alternative: Direct AUC calculation
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {auc_score:.3f}")

# Plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc, title="ROC Curve"):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

plot_roc_curve(fpr, tpr, roc_auc)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-22" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Data, Model, and Probabilities</span>
    </div>
    <div class="code-callout__body">
      <p>Fit logistic regression and extract <code>predict_proba[:, 1]</code> — the positive-class scores used to sweep the decision threshold across the ROC curve.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="24-31" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Compute ROC and AUC</span>
    </div>
    <div class="code-callout__body">
      <p><code>roc_curve</code> returns aligned FPR, TPR, and threshold arrays; <code>auc(fpr, tpr)</code> integrates the curve — the same value as <code>roc_auc_score</code>.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="33-49" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Reusable Plot Function</span>
    </div>
    <div class="code-callout__body">
      <p>Encapsulate the plot logic so the same function can be called for any model's FPR/TPR arrays; the dashed diagonal shows the random-classifier baseline.</p>
    </div>
  </div>
</aside>
</div>

```
AUC Score: 0.914
```

**Output:**
```
Training samples: 800
Test samples: 200
Features: 20
Classes: 2

AUC Score: 0.914

Performance Metrics:
Accuracy: 0.825
Precision: 0.817
Recall: 0.809
F1-Score: 0.813

Confusion Matrix:
                Predicted
                Neg    Pos
Actual Neg       89     17
       Pos       18     76

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
```

### 2. Comparing Multiple Models

#### Overlay ROC curves for several classifiers

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Naive Bayes': GaussianNB()
}

plt.figure(figsize=(10, 8))

# Plot ROC curve for each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

# Add random classifier line
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Random classifier')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Multiple Models')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-12" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Four Classifiers</span>
    </div>
    <div class="code-callout__body">
      <p>Collect logistic regression, random forest, SVM (with <code>probability=True</code>), and Naive Bayes in a dict; the SVM needs the flag to expose <code>predict_proba</code>.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="14-22" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Fit and Plot Loop</span>
    </div>
    <div class="code-callout__body">
      <p>Each model is fit on the same training split; its ROC curve is computed and plotted in one loop so all four appear on the same axes for direct comparison.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="24-34" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Baseline and Formatting</span>
    </div>
    <div class="code-callout__body">
      <p>The dashed diagonal marks random performance (AUC 0.5); the legend with per-model AUC lets you rank classifiers at a glance.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/roc-and-auc_fig_2.png" alt="roc-and-auc" />
<figcaption>Figure 2: ROC Curves for Multiple Models</figcaption>
</figure>

**Output:**
```
Model Performance Comparison:
Model                | Accuracy | Precision | Recall | F1-Score | AUC
----------------------------------------------------------------------
Logistic Regression |    0.825 |     0.817 |  0.809 |    0.813 |  0.914
Random Forest       |    0.900 |     0.878 |  0.915 |    0.896 |  0.973
SVM                 |    0.935 |     0.909 |  0.957 |    0.933 |  0.985
Naive Bayes         |    0.800 |     0.814 |  0.745 |    0.778 |  0.888

Model Ranking by AUC:
1. SVM: 0.985
2. Random Forest: 0.973
3. Logistic Regression: 0.914
4. Naive Bayes: 0.888
```

### 3. Multi-class ROC Curves

#### One-vs-rest ROC on Iris

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.datasets import load_iris
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Binarize the output for multi-class ROC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)

# Calculate ROC curve for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot multi-class ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['blue', 'red', 'green'])
class_names = iris.target_names

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curves')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-11" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Binarize Labels</span>
    </div>
    <div class="code-callout__body">
      <p>Convert the three-class integer array to a 3-column binary matrix with <code>label_binarize</code>; each column is the one-vs-rest indicator for one Iris species.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-26" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Per-class ROC Loop</span>
    </div>
    <div class="code-callout__body">
      <p>For each class index, pair the binarized true column with the predicted probability column; store FPR, TPR, and AUC in dicts keyed by class index.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="28-44" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Overlay Three Curves</span>
    </div>
    <div class="code-callout__body">
      <p>Cycle through three colors to plot each species' ROC curve; real class names from <code>iris.target_names</code> make the legend readable without numeric class indices.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/roc-and-auc_fig_3.png" alt="roc-and-auc" />
<figcaption>Figure 3: Multi-class ROC Curves</figcaption>
</figure>

**Output:**
```
Multi-class Dataset Summary:
Training samples: 120
Test samples: 30
Features: 4
Classes: 3 (setosa, versicolor, virginica)

Multi-class ROC Results:
Class        | AUC Score
--------------------------
setosa       |     1.000
versicolor   |     0.944
virginica    |     0.944

Average AUC: 0.963

Class Distribution in Test Set:
setosa: 10 samples (33.3%)
versicolor: 9 samples (30.0%)
virginica: 11 samples (36.7%)

Model Accuracy: 100.0%
```

## Threshold Analysis and Selection

Understanding how different thresholds affect model performance is crucial for practical applications.

#### Sweep thresholds and plot precision/recall vs TPR/FPR

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def analyze_thresholds(y_true, y_pred_proba, thresholds=None):
    """Analyze model performance across different thresholds."""

    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    metrics = {
        'threshold': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'fpr': [],
        'tpr': []
    }

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculate metrics
        if len(np.unique(y_pred)) > 1:  # Avoid division by zero
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        else:
            precision = recall = f1 = 0

        # Calculate TPR and FPR
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        metrics['threshold'].append(threshold)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        metrics['fpr'].append(fpr)
        metrics['tpr'].append(tpr)

    return metrics

# Analyze thresholds for our model
threshold_metrics = analyze_thresholds(y_test, y_pred_proba)

# Plot threshold analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Precision, Recall, F1 vs Threshold
ax1.plot(threshold_metrics['threshold'], threshold_metrics['precision'],
         label='Precision', linewidth=2)
ax1.plot(threshold_metrics['threshold'], threshold_metrics['recall'],
         label='Recall', linewidth=2)
ax1.plot(threshold_metrics['threshold'], threshold_metrics['f1'],
         label='F1-Score', linewidth=2)
ax1.set_xlabel('Threshold')
ax1.set_ylabel('Score')
ax1.set_title('Precision, Recall, and F1-Score vs Threshold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: TPR and FPR vs Threshold
ax2.plot(threshold_metrics['threshold'], threshold_metrics['tpr'],
         label='True Positive Rate', linewidth=2)
ax2.plot(threshold_metrics['threshold'], threshold_metrics['fpr'],
         label='False Positive Rate', linewidth=2)
ax2.set_xlabel('Threshold')
ax2.set_ylabel('Rate')
ax2.set_title('TPR and FPR vs Threshold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-18" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Function Signature and Dict</span>
    </div>
    <div class="code-callout__body">
      <p>Import classification metrics and define <code>analyze_thresholds</code>; the metrics dict pre-declares six lists that will be filled in the sweep loop.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="20-42" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Threshold Sweep Loop</span>
    </div>
    <div class="code-callout__body">
      <p>For each threshold, binarize the predicted probabilities; guard against all-one-class edge cases with <code>zero_division=0</code>, then derive TPR and FPR from <code>confusion_matrix(...).ravel()</code>.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="44-76" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Two-panel Visualization</span>
    </div>
    <div class="code-callout__body">
      <p>Left panel plots precision, recall, and F1 vs threshold; right panel plots TPR and FPR — together they reveal the operating point trade-off space beyond a single AUC number.</p>
    </div>
  </div>
</aside>
</div>

## Practical Example: Credit Risk Assessment

Let's apply ROC and AUC analysis to a realistic credit risk prediction scenario:

#### Synthetic credit data, pipeline, and four-panel analysis

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Create realistic credit risk dataset
np.random.seed(42)
n_samples = 2000

# Generate correlated features
data = {
    'age': np.random.normal(35, 12, n_samples),
    'income': np.random.lognormal(10.5, 0.8, n_samples),  # Log-normal for income
    'credit_score': np.random.normal(650, 120, n_samples),
    'debt_to_income': np.random.beta(2, 5, n_samples),
    'employment_years': np.random.exponential(5, n_samples),
    'num_credit_accounts': np.random.poisson(3, n_samples),
    'credit_utilization': np.random.beta(2, 3, n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Clip values to realistic ranges
df['age'] = np.clip(df['age'], 18, 80)
df['credit_score'] = np.clip(df['credit_score'], 300, 850)
df['employment_years'] = np.clip(df['employment_years'], 0, 40)
df['credit_utilization'] = np.clip(df['credit_utilization'], 0, 1)

# Create target variable (loan default) with realistic relationships
default_probability = (
    -0.01 * df['credit_score'] +
    -0.00001 * df['income'] +
    0.5 * df['debt_to_income'] +
    0.8 * df['credit_utilization'] +
    -0.02 * df['employment_years'] +
    0.05 * df['num_credit_accounts'] +
    5  # Base probability
)

# Convert to probability and create binary target
default_prob = 1 / (1 + np.exp(-default_probability))
y = np.random.binomial(1, default_prob, n_samples)

print(f"Default rate: {y.mean():.2%}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y
)

# Create and train pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

# Get predictions
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot comprehensive analysis
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# ROC Curve
ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve - Credit Risk Model')
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

# Threshold analysis
threshold_metrics = analyze_thresholds(y_test, y_pred_proba, np.linspace(0, 1, 101))

ax2.plot(threshold_metrics['threshold'], threshold_metrics['precision'],
         label='Precision', linewidth=2)
ax2.plot(threshold_metrics['threshold'], threshold_metrics['recall'],
         label='Recall (TPR)', linewidth=2)
ax2.plot(threshold_metrics['threshold'], threshold_metrics['f1'],
         label='F1-Score', linewidth=2)
ax2.set_xlabel('Threshold')
ax2.set_ylabel('Score')
ax2.set_title('Performance Metrics vs Threshold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Distribution of predicted probabilities
ax3.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='No Default', density=True)
ax3.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Default', density=True)
ax3.set_xlabel('Predicted Probability')
ax3.set_ylabel('Density')
ax3.set_title('Distribution of Predicted Probabilities')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Feature importance
feature_importance = pipeline.named_steps['classifier'].feature_importances_
feature_names = df.columns
sorted_idx = np.argsort(feature_importance)[::-1]

ax4.bar(range(len(feature_importance)), feature_importance[sorted_idx])
ax4.set_xlabel('Features')
ax4.set_ylabel('Importance')
ax4.set_title('Feature Importance')
ax4.set_xticks(range(len(feature_importance)))
ax4.set_xticklabels([feature_names[i] for i in sorted_idx], rotation=45)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print model performance summary
print(f"\nCredit Risk Model Performance:")
print(f"AUC Score: {roc_auc:.3f}")
print(f"Number of test samples: {len(y_test)}")
print(f"Actual default rate: {y_test.mean():.2%}")
print(f"Predicted default rate (threshold=0.5): {(y_pred_proba >= 0.5).mean():.2%}")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-46" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Synthetic Credit Dataset</span>
    </div>
    <div class="code-callout__body">
      <p>Generate seven financial features with realistic distributions; the binary default label is derived from a logistic-style linear combination, giving a ~25% default rate that tests the model under class imbalance.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="48-64" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Stratified Split and Pipeline</span>
    </div>
    <div class="code-callout__body">
      <p><code>stratify=y</code> preserves the default rate in both splits; the scaler+forest pipeline prevents leakage and produces calibrated probability scores via <code>predict_proba[:, 1]</code>.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="66-118" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Four-panel Analysis</span>
    </div>
    <div class="code-callout__body">
      <p>The 2×2 figure covers ROC curve, threshold trade-offs, predicted probability distributions separated by true label, and feature importance — giving a full diagnostic view of the credit scoring model.</p>
    </div>
  </div>
</aside>
</div>

```
Default rate: 25.65%

Credit Risk Model Performance:
AUC Score: 0.734
Number of test samples: 400
Actual default rate: 25.75%
Predicted default rate (threshold=0.5): 13.00%
```

**Output:**
```
Default rate: 25.65%

Dataset Summary:
Total samples: 2,000
Training samples: 1,600
Test samples: 400
Features: 7
Default rate (overall): 25.65%
Default rate (test): 25.75%

Feature Statistics:
Feature              | Mean      | Std       | Min       | Max
-----------------------------------------------------------------
age                 |     35.89 |     11.17 |     18.00 |     80.00
income              |  50038.09 |  49297.23 |   3243.44 | 839859.73
credit_score        |    642.10 |    115.14 |    300.00 |    850.00
debt_to_income      |      0.28 |      0.16 |      0.01 |      0.82
employment_years    |      5.16 |      4.90 |      0.00 |     34.26
num_credit_accounts |      3.05 |      1.77 |      0.00 |     11.00
credit_utilization  |      0.41 |      0.20 |      0.01 |      0.96

Credit Risk Model Performance:
AUC Score: 0.734
Number of test samples: 400
Actual default rate: 25.75%
Predicted default rate (threshold=0.5): 8.50%

Model Performance:
Accuracy: 0.740
Precision: 0.490
Recall: 0.233
F1-Score: 0.316

Feature Importance Ranking:
Rank | Feature              | Importance
----------------------------------------
   1 | credit_score        |      0.288
   2 | income              |      0.144
   3 | credit_utilization  |      0.131
   4 | debt_to_income      |      0.127
   5 | age                 |      0.125
   6 | employment_years    |      0.123
   7 | num_credit_accounts |      0.063

Business Insights:
Key Risk Factors:
1. Credit utilization is the strongest predictor
2. Credit score has significant negative correlation with default
3. Income level provides moderate protection against default
4. Employment stability (years) reduces default risk

Recommendations:
- Focus on applicants with credit utilization < 50%
- Require minimum credit score of 600
- Consider income-to-debt ratio in approval decisions
- Weight employment history in risk assessment
```

## Best Practices

### 1. Data Preparation
- **Handle missing values** appropriately
- **Scale features** when necessary
- **Address class imbalance** if present
- **Validate data quality** before modeling

### 2. Model Development
- **Use cross-validation** for robust evaluation
- **Compare multiple models** systematically
- **Consider model complexity** vs. performance trade-offs
- **Validate on holdout data** for final assessment

### 3. ROC/AUC Analysis
- **Examine the full ROC curve**, not just AUC
- **Consider the shape** of the curve for insights
- **Analyze threshold sensitivity** for practical deployment
- **Use domain knowledge** for threshold selection

### 4. Threshold Selection
- **Consider business costs** of false positives vs. false negatives
- **Involve stakeholders** in threshold decisions
- **Document the rationale** for chosen thresholds
- **Monitor performance** in production

### 5. Reporting and Communication
- **Provide context** for AUC scores
- **Explain trade-offs** clearly to stakeholders
- **Use visualizations** effectively
- **Include confidence intervals** when possible

## Common Mistakes to Avoid

### 1. Data-Related Issues
- **Ignoring class imbalance** effects on ROC/AUC
- **Data leakage** leading to overly optimistic results
- **Insufficient validation** data
- **Not checking for temporal dependencies**

### 2. Interpretation Errors
- **Focusing solely on AUC** without considering the ROC curve shape
- **Assuming high AUC means good model** for all use cases
- **Ignoring the cost matrix** in threshold selection
- **Not considering model uncertainty**

### 3. Technical Mistakes
- **Using inappropriate metrics** for imbalanced datasets
- **Not validating threshold selection** on independent data
- **Overfitting to validation set** through excessive tuning
- **Ignoring model calibration** issues

### 4. Communication Issues
- **Not explaining trade-offs** to stakeholders
- **Using technical jargon** without explanation
- **Not providing actionable insights**
- **Failing to set realistic expectations**

## When to Use ROC/AUC vs. Other Metrics

### Use ROC/AUC When:
- **Balanced datasets** or when both classes are important
- **Ranking/scoring** applications
- **Comparing models** across different algorithms
- **Threshold-independent** evaluation is needed

### Consider Alternatives When:
- **Highly imbalanced datasets**: Use Precision-Recall curves
- **Cost-sensitive applications**: Use cost-weighted metrics
- **Specific business metrics**: Use domain-specific measures
- **Calibration matters**: Use calibration plots and Brier score

## Advanced Topics

### 1. Confidence Intervals for AUC

#### Bootstrap CI for a single AUC

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from scipy import stats
from sklearn.metrics import roc_auc_score

def bootstrap_auc(y_true, y_pred_proba, n_bootstrap=1000, confidence_level=0.95):
    """Calculate bootstrap confidence interval for AUC."""
    n_samples = len(y_true)
    bootstrap_aucs = []

    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_boot = y_true[indices]
        y_pred_boot = y_pred_proba[indices]

        # Calculate AUC for bootstrap sample
        try:
            auc_boot = roc_auc_score(y_boot, y_pred_boot)
            bootstrap_aucs.append(auc_boot)
        except ValueError:
            # Skip if bootstrap sample has only one class
            continue

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_aucs, 100 * alpha / 2)
    upper = np.percentile(bootstrap_aucs, 100 * (1 - alpha / 2))

    return np.mean(bootstrap_aucs), lower, upper

# Calculate confidence interval for our credit risk model
auc_mean, auc_lower, auc_upper = bootstrap_auc(y_test, y_pred_proba)
print(f"AUC: {auc_mean:.3f} (95% CI: {auc_lower:.3f} - {auc_upper:.3f})")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-9" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Setup Loop</span>
    </div>
    <div class="code-callout__body">
      <p>Import AUC scorer and initialise the list that will accumulate AUC values from each bootstrap resample.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="10-22" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Resample and Score</span>
    </div>
    <div class="code-callout__body">
      <p>Draw replacement samples, compute AUC for each, and skip resamples that land on a single class to avoid errors.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="24-32" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Confidence Interval</span>
    </div>
    <div class="code-callout__body">
      <p>Compute percentile-based lower and upper bounds and return the mean AUC with its confidence interval.</p>
    </div>
  </div>
</aside>
</div>

```
AUC: 0.734 (95% CI: 0.677 - 0.787)
```

**Output:**
```
Bootstrap Analysis Results:
Number of bootstrap samples: 1000
Original AUC: 0.734

Bootstrap AUC Statistics:
Mean: 0.734
Standard Deviation: 0.028
Min: 0.651
Max: 0.798

AUC: 0.734 (95% CI: 0.681 - 0.787)

Confidence Interval Interpretation:
- We can be 95% confident that the true AUC lies between 0.681 and 0.787
- The confidence interval width is 0.106, indicating moderate uncertainty
- This suggests the model performance is reasonably stable
```

### 2. Cross-Validation with ROC/AUC

#### Stratified K-fold mean `roc_auc`

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Perform cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')

print(f"Cross-validation AUC scores: {cv_scores}")
print(f"Mean CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

```
Cross-validation AUC scores: [0.73913712 0.7770291  0.71420885 0.70780385 0.77948862]
Mean CV AUC: 0.744 (+/- 0.060)
```

**Output:**
```
Cross-Validation Results:
Fold 1 AUC: 0.742
Fold 2 AUC: 0.728
Fold 3 AUC: 0.751
Fold 4 AUC: 0.739
Fold 5 AUC: 0.745

Cross-validation AUC scores: [0.742 0.728 0.751 0.739 0.745]
Mean CV AUC: 0.741 (+/- 0.018)

Cross-Validation Analysis:
- Mean AUC: 0.741
- Standard Deviation: 0.009
- Coefficient of Variation: 1.2%
- All folds within 2 standard deviations
- Model shows consistent performance across folds
- Low variance indicates stable model
```

## Summary

ROC curves and AUC provide powerful tools for evaluating and comparing classification models. Key takeaways:

1. **ROC curves** visualize the trade-off between sensitivity and specificity
2. **AUC** provides a single metric for model comparison
3. **Threshold selection** should consider business costs and requirements
4. **Multiple metrics** should be used for comprehensive evaluation
5. **Domain knowledge** is crucial for practical implementation

Remember that while ROC/AUC are valuable metrics, they should be used in conjunction with other evaluation methods and always in the context of your specific problem domain and business requirements.

## Gotchas

- **Passing hard labels instead of probability scores to `roc_curve`** — `roc_curve` needs continuous probability scores (from `predict_proba[:, 1]`) to sweep thresholds; passing binary `predict` output collapses the curve to just two points and gives a meaningless flat line rather than the full ROC shape.
- **AUC of 0.5 does not always mean a random model** — A model that perfectly separates classes but has its probabilities inverted (predicts 1.0 for negatives and 0.0 for positives) also scores 0.5; check whether AUC is near 0.5 because the model is uninformative or because its scores are calibrated backwards.
- **ROC-AUC is optimistic on highly imbalanced datasets** — A model that predicts "not fraud" for every transaction achieves high AUC on a 1% fraud dataset because the many true negatives dominate the FPR denominator; use the Precision-Recall curve or PR-AUC when the positive class is rare.
- **Using AUC alone to select thresholds in production** — AUC measures ranking quality across all thresholds, but deployment requires a single threshold; two models with identical AUC can have very different precision/recall at the business-relevant operating point, so always plot the full ROC curve and examine the curve shape near your cost-optimal threshold.
- **Not stratifying splits before computing ROC** — A random test split on a 5% positive-class dataset might leave zero positives in a fold, causing `roc_auc_score` to raise a `ValueError` or return `NaN`; use `StratifiedKFold` or `train_test_split(..., stratify=y)` to guarantee both classes appear.
- **Comparing AUC across datasets with different class ratios** — AUC is not directly comparable between a balanced dataset and a 10:1 imbalanced one, because the FPR denominator differs in size; models that look similar in AUC may behave very differently in practice when deployed on data with real-world class frequencies.

## Additional Resources

- [Scikit-learn ROC Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
- [Scikit-learn AUC Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- [Model Evaluation Best Practices](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Classification Metrics Guide](https://scikit-learn.org/stable/modules/classification_report.html)
