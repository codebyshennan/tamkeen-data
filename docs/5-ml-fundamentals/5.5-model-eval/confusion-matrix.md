---
reading_minutes: 14
objectives:
  - "Read a 2×2 confusion matrix (TP, FP, TN, FN) and extend the same idea to multi-class — each off-diagonal cell tells you which classes the model confuses."
  - "Generate one with `sklearn.metrics.confusion_matrix` and visualise it with `ConfusionMatrixDisplay` (or a seaborn heatmap)."
  - "Derive accuracy, precision, recall, and F1 from the matrix instead of treating those metrics as separate computations."
  - "Choose row vs column normalisation deliberately so percentages answer the question you actually have (per-true-class recall vs per-predicted-class precision)."
---

# Confusion Matrix

**After this lesson:** you can explain the core ideas in “Confusion Matrix” and reproduce the examples here in your own notebook or environment.

## Overview

Builds the **TP/TN/FP/FN** grid so precision, recall, and related rates have a shared ground truth.

## Introduction

A confusion matrix is a fundamental tool in machine learning for evaluating classification models. It provides a detailed breakdown of model predictions versus actual values, helping to understand model performance across different classes.

### Video Tutorial: Confusion Matrix Explained

<div class="video-embed">
<iframe width="560" height="315" src="https://www.youtube.com/embed/Kdsp6soqA7o" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

*StatQuest: Machine Learning Fundamentals: The Confusion Matrix by Josh Starmer*

## What is a Confusion Matrix?

A confusion matrix is a table that describes the performance of a classification model by comparing predicted values with actual values. Think of it as a "report card" for your model that shows exactly where it's getting things right and wrong.

![Confusion Matrix](assets/confusion_matrix.png)

The confusion matrix shows four key components for binary classification:

- **True Positives (TP)**: Correctly predicted positive cases - "We said YES, and it was YES"
- **True Negatives (TN)**: Correctly predicted negative cases - "We said NO, and it was NO"  
- **False Positives (FP)**: Incorrectly predicted positive cases - "We said YES, but it was NO" (Type I Error)
- **False Negatives (FN)**: Incorrectly predicted negative cases - "We said NO, but it was YES" (Type II Error)

### Real-World Example: Medical Diagnosis

Imagine a model predicting whether a patient has a disease:

| | Predicted: No Disease | Predicted: Disease |
|---|---|---|
| **Actual: No Disease** | TN: Healthy person correctly identified | FP: Healthy person incorrectly diagnosed |
| **Actual: Disease** | FN: Sick person missed (dangerous!) | TP: Sick person correctly identified |

**Why this matters:**
- **False Negatives (FN)**: Missing a sick patient could be life-threatening
- **False Positives (FP)**: Unnecessary worry and treatment for healthy patients
- The cost of each error type is different!

## Types of Confusion Matrices

### 1. Binary Classification

#### Logistic regression + heatmap

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Get predictions
y_pred = model.predict(X_test)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
{% endhighlight %}

<figure>
<img src="assets/confusion-matrix_fig_1.png" alt="confusion-matrix" />
<figcaption>Figure 1: Confusion Matrix</figcaption>
</figure>

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports and Data</span>
    </div>
    <div class="code-callout__body">
      <p>Generate a synthetic 1000-sample binary classification dataset with 15 informative features, then split 80/20 into train and test sets.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="10-21" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Train and Predict</span>
    </div>
    <div class="code-callout__body">
      <p>Fit logistic regression, generate test-set predictions, then compute the 2×2 confusion matrix comparing true vs predicted labels.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="23-29" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Heatmap Visualization</span>
    </div>
    <div class="code-callout__body">
      <p><code>sns.heatmap</code> with <code>annot=True</code> writes raw counts in each cell; rows are true labels, columns are predicted labels.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/confusion-matrix_fig_1.png" alt="confusion-matrix" />
<figcaption>Figure 1: Confusion Matrix</figcaption>
</figure>

### 2. Multi-class Classification

#### Iris: 3×3 confusion matrix

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get predictions
y_pred = model.predict(X_test)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Multi-class Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
{% endhighlight %}

<figure>
<img src="assets/confusion-matrix_fig_2.png" alt="confusion-matrix" />
<figcaption>Figure 2: Multi-class Confusion Matrix</figcaption>
</figure>

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Load Iris and Split</span>
    </div>
    <div class="code-callout__body">
      <p>The Iris dataset has three classes; the same <code>confusion_matrix</code> API produces a 3×3 matrix without any changes to the call.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="10-17" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Train Random Forest and Predict</span>
    </div>
    <div class="code-callout__body">
      <p>A Random Forest classifier is fit on train data; predictions and the resulting confusion matrix reveal which species pairs are most confused.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="19-25" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Multi-class Heatmap</span>
    </div>
    <div class="code-callout__body">
      <p>The 3×3 heatmap shows diagonal hits and off-diagonal confusions — off-diagonal entries reveal which class pairs the model struggles to separate.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/confusion-matrix_fig_2.png" alt="confusion-matrix" />
<figcaption>Figure 2: Multi-class Confusion Matrix</figcaption>
</figure>

## Interpreting Confusion Matrices

### 1. Binary Classification

- True Positives (TP): Correctly identified positive cases
- True Negatives (TN): Correctly identified negative cases
- False Positives (FP): Type I errors
- False Negatives (FN): Type II errors

### 2. Multi-class Classification

- Diagonal elements: Correct predictions
- Off-diagonal elements: Misclassifications
- Row sums: Actual class distribution
- Column sums: Predicted class distribution

### 3. Performance Metrics

- Accuracy: (TP + TN) / (TP + TN + FP + FN)
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1 Score: 2 *(Precision* Recall) / (Precision + Recall)

## Best Practices

1. **Choose Appropriate Visualization**
   - Clear labels and title
   - Proper color scheme
   - Informative annotations
   - Grid lines

2. **Consider Class Imbalance**
   - Use appropriate metrics
   - Consider cost-sensitive learning
   - Apply class weighting

3. **Interpret Results Carefully**
   - Look for patterns
   - Identify systematic errors
   - Consider business impact

4. **Use Multiple Metrics**
   - Don't rely on accuracy alone
   - Consider precision and recall
   - Use F1 score for balance

## Common Mistakes to Avoid

1. **Ignoring Class Imbalance**
   - Using raw counts
   - Not considering costs
   - Missing important patterns

2. **Poor Visualization**
   - Unclear labels
   - Wrong color scheme
   - Missing context

3. **Misinterpretation**
   - Focusing on wrong metrics
   - Ignoring business impact
   - Overlooking patterns

## Practical Example: Credit Risk Prediction

Let's analyze a confusion matrix for a credit risk prediction model:

#### Synthetic credit features + pipeline + CM

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Create credit risk dataset
np.random.seed(42)
n_samples = 1000

# Generate features
data = {
    'age': np.random.normal(35, 10, n_samples),
    'income': np.random.exponential(50000, n_samples),
    'credit_score': np.random.normal(700, 100, n_samples),
    'debt_ratio': np.random.beta(2, 5, n_samples),
    'employment_length': np.random.exponential(5, n_samples)
}

X = pd.DataFrame(data)
y = (X['credit_score'] + X['income']/1000 + X['age'] > 800).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# Get predictions
y_pred = pipeline.predict(X_test)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Credit Risk Prediction')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
{% endhighlight %}

<figure>
<img src="assets/confusion-matrix_fig_3.png" alt="confusion-matrix" />
<figcaption>Figure 3: Confusion Matrix for Credit Risk Prediction</figcaption>
</figure>

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-9" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports</span>
    </div>
    <div class="code-callout__body">
      <p>Import pandas, sklearn pipeline components, and seaborn — everything needed to build, train, and visualize a real-world credit pipeline.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="11-26" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Synthetic Credit Data</span>
    </div>
    <div class="code-callout__body">
      <p>Generate 1000 applicants with realistic financial distributions; the label is a simple threshold rule on score, income, and age.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="28-40" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Pipeline and Training</span>
    </div>
    <div class="code-callout__body">
      <p>A <code>Pipeline</code> chains <code>StandardScaler</code> and <code>RandomForestClassifier</code> so scaling is learned only from training data, preventing leakage.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="42-51" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Confusion Matrix Heatmap</span>
    </div>
    <div class="code-callout__body">
      <p>Compute and plot the binary confusion matrix with business-relevant labels (approve/reject) to reveal approval misclassification costs.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/confusion-matrix_fig_3.png" alt="confusion-matrix" />
<figcaption>Figure 3: Confusion Matrix for Credit Risk Prediction</figcaption>
</figure>

## Gotchas

- **Row vs column orientation** — sklearn's `confusion_matrix` places **true labels on rows** and **predicted labels on columns** (the standard mathematical convention), but some plotting libraries and papers swap these axes; always label the heatmap explicitly and verify which axis is which before reading FP and FN counts.
- **Raw counts mislead on imbalanced data** — A matrix showing 950 TNs and 5 TPs might look acceptable, but the model is nearly useless if there are 45 actual positives; always compute normalized rates (precision, recall) alongside raw counts when class sizes differ significantly.
- **`sklearn.metrics.plot_confusion_matrix` is deprecated** — It was removed in sklearn 1.2; use `ConfusionMatrixDisplay.from_estimator` or `ConfusionMatrixDisplay.from_predictions` instead, otherwise your code silently breaks on updated environments.
- **Multi-class off-diagonals require row-wise reading** — In an N×N matrix, row *i* shows all predictions for true class *i*; an off-diagonal at row 1, column 2 means true class 1 was predicted as class 2, not the reverse; confusing direction leads to misidentifying which classes are confused.
- **Comparing confusion matrices across differently-sized test sets** — Absolute counts from a 200-sample test set cannot be compared directly to counts from a 2000-sample set; normalize by row (`normalize='true'` in sklearn) to compare recall rates on a level playing field.

## Additional Resources

1. Scikit-learn documentation on confusion matrices
2. Research papers on classification metrics
3. Online tutorials on model evaluation
4. Books on machine learning evaluation
