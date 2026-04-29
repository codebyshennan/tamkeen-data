---
reading_minutes: 12
objectives:
  - "Define accuracy as the fraction of correct predictions and compute it for binary and multi-class problems with `sklearn.metrics.accuracy_score`."
  - "Recognise when accuracy is misleading: with class imbalance (>~80% majority), report at least one of precision/recall, F1, or a confusion matrix alongside it."
  - "Compare your model's accuracy to the **majority-class baseline** before celebrating — beating \"always predict the most common label\" is the actual bar."
  - "Validate with cross-validation and a held-out test set, not the training set, to avoid mistaking memorisation for skill."
---

# Accuracy

**After this lesson:** you can explain the core ideas in “Accuracy” and reproduce the examples here in your own notebook or environment.

## Overview

**Accuracy** as a baseline fraction correct—when it is meaningful and when **class imbalance** makes it misleading.

Relates to [confusion matrix](confusion-matrix.md) and [metrics](metrics.md).

## Helpful video

StatQuest: why cross-validation matters for model evaluation.

<iframe width="560" height="315" src="https://www.youtube.com/embed/fSytzGwwBVw" title="Machine Learning Fundamentals: Cross Validation" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Introduction

Accuracy is one of the most fundamental metrics in machine learning, measuring the proportion of correct predictions made by a model. While simple to understand and calculate, it's important to use accuracy appropriately and understand its limitations.

## What is Accuracy?

Accuracy is the ratio of correct predictions to total predictions:

\\[
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
\\]

{% include mermaid-diagram.html src="5-ml-fundamentals/5.5-model-eval/diagrams/accuracy-1.mmd" %}

*Rule of thumb: if your dataset has < 80% majority class, accuracy is still OK as a sanity check. Beyond that, always report at least one additional metric.*

## Types of Accuracy

### 1. Binary Classification

#### Compute accuracy on a held-out set (binary)

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
from sklearn.metrics import accuracy_score
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

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
{% endhighlight %}
```
Accuracy: 0.810
```

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-10" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Data and Split</span>
    </div>
    <div class="code-callout__body">
      <p>Generate a balanced binary classification dataset and split 80/20; the class balance here makes accuracy a meaningful baseline metric.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="12-20" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Train and Score</span>
    </div>
    <div class="code-callout__body">
      <p>Fit logistic regression, call <code>predict</code> for hard labels, then pass both to <code>accuracy_score</code>—the fraction of matching indices across all test samples.</p>
    </div>
  </div>
</aside>
</div>

### 2. Multi-class Classification

#### Accuracy with three classes (Iris)

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
{% endhighlight %}
```
Accuracy: 1.000
```

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-10" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Iris Setup</span>
    </div>
    <div class="code-callout__body">
      <p>Load the three-class Iris dataset and split into train/test; with only 150 samples, <code>test_size=0.2</code> reserves 30 samples for evaluation.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="12-20" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Random Forest Accuracy</span>
    </div>
    <div class="code-callout__body">
      <p>Fit a Random Forest and measure accuracy; for well-separated Iris classes this typically reaches 1.0, illustrating that accuracy is reliable when classes are balanced and separable.</p>
    </div>
  </div>
</aside>
</div>

## Interpreting Accuracy

### 1. Binary Classification

- Perfect accuracy: 1.0
- Random guessing: 0.5
- Worst case: 0.0

### 2. Multi-class Classification

- Perfect accuracy: 1.0
- Random guessing: 1/n_classes
- Worst case: 0.0

### 3. Balanced vs. Imbalanced Data

- Balanced: Accuracy is meaningful
- Imbalanced: May be misleading
- Consider other metrics

## Best Practices

1. **Choose Appropriate Metrics**
   - Consider class distribution
   - Use multiple metrics
   - Look at confusion matrix

2. **Handle Class Imbalance**
   - Use weighted accuracy
   - Consider other metrics
   - Apply resampling

3. **Validate Results**
   - Use cross-validation
   - Check for overfitting
   - Compare with baseline

4. **Consider Business Impact**
   - Cost of errors
   - Risk tolerance
   - Decision thresholds

## Common Mistakes to Avoid

1. **Relying Solely on Accuracy**
   - Ignoring class imbalance
   - Missing important patterns
   - Overlooking costs

2. **Poor Data Preparation**
   - Not handling missing values
   - Ignoring outliers
   - Skipping preprocessing

3. **Incorrect Interpretation**
   - Not considering baseline
   - Ignoring business context
   - Overlooking costs

## Practical Example: Credit Risk Prediction

Let's analyze accuracy for a credit risk prediction model:

#### Pipeline accuracy vs majority baseline

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Calculate baseline accuracy
baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())
print(f"Baseline Accuracy: {baseline_accuracy:.3f}")
{% endhighlight %}
```
Accuracy: 0.970
Baseline Accuracy: 0.555
```

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-24" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Credit Dataset</span>
    </div>
    <div class="code-callout__body">
      <p>Generate five financial features and derive a binary approval label from a threshold on credit score, income, and age — the same synthetic credit setup reused across 5.5 examples.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="26-40" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Pipeline and Prediction</span>
    </div>
    <div class="code-callout__body">
      <p>A scaler+forest pipeline prevents data leakage; <code>predict</code> returns hard labels used to compute the test-set accuracy score.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="42-48" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Accuracy vs Baseline</span>
    </div>
    <div class="code-callout__body">
      <p>Report both model accuracy and the majority-class baseline (<code>max(P(y=1), P(y=0))</code>); the gap between the two shows how much the model actually learns beyond trivial prediction.</p>
    </div>
  </div>
</aside>
</div>

```
Accuracy: 0.970
Baseline Accuracy: 0.555
```

## Gotchas

- **High accuracy on an imbalanced dataset feels like success** — A dataset with 95% negative samples lets a classifier that always predicts "negative" achieve 95% accuracy; this is the accuracy paradox, and the model has learned nothing; always compare model accuracy to the majority-class baseline before declaring success.
- **`model.score(X_test, y_test)` returns accuracy by default for classifiers** — This is easy to overlook when you later switch to a regression problem where `.score` returns R², not MSE; explicitly call `accuracy_score` or the relevant metric function to make your intent clear and avoid silent metric changes.
- **Balanced accuracy is not the same as accuracy on balanced data** — `sklearn.metrics.balanced_accuracy_score` averages recall per class and is appropriate for imbalanced data; it will differ from standard accuracy even on a balanced dataset if per-class recalls are unequal; choose the right function deliberately.
- **Comparing accuracy across different test set sizes** — Accuracy from a 50-sample test is far noisier than accuracy from a 5000-sample test; a 2% gap between two models may be statistically insignificant on 50 samples; always report confidence intervals or use statistical tests when comparing models on small test sets.
- **Using accuracy as the scoring metric inside `GridSearchCV` for imbalanced problems** — Setting `scoring='accuracy'` in `GridSearchCV` selects the model that maximises accuracy, which on imbalanced data rewards the model that best predicts the majority class; use `scoring='f1'`, `'roc_auc'`, or a custom scorer aligned with your actual objective.

## Additional Resources

1. Scikit-learn documentation on accuracy metrics
2. Research papers on classification metrics
3. Online tutorials on model evaluation
4. Books on machine learning evaluation
