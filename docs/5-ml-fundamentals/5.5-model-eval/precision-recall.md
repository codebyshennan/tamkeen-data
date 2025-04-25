# Precision and Recall

## Introduction

Precision and Recall are fundamental metrics in machine learning for evaluating classification models. They provide insights into a model's performance in terms of accuracy and completeness.

## What are Precision and Recall?

### Precision

- Definition: Ratio of true positives to all predicted positives
- Formula: TP / (TP + FP)
- Interpretation: How many of the predicted positive cases are actually positive
- Range: 0 to 1 (higher is better)

### Recall

- Definition: Ratio of true positives to all actual positives
- Formula: TP / (TP + FN)
- Interpretation: How many of the actual positive cases are correctly identified
- Range: 0 to 1 (higher is better)

## Types of Precision-Recall Curves

### 1. Binary Classification

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Get prediction probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
average_precision = average_precision_score(y_test, y_pred_proba)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()
```

### 2. Multi-class Classification

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score
from itertools import cycle

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Binarize the output
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get prediction probabilities
y_pred_proba = model.predict_proba(X_test)

# Calculate precision-recall curve for each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])

# Plot precision-recall curves
plt.figure(figsize=(8, 6))
colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label=f'Precision-Recall curve of class {i} (AP = {average_precision[i]:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Multi-class Precision-Recall Curves')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()
```

## Interpreting Precision-Recall Curves

### 1. Binary Classification

- Area Under Curve (AUC): Overall model performance
- Perfect classifier: AUC = 1.0
- Random classifier: AUC = 0.5
- Good classifier: AUC > 0.8
- Poor classifier: AUC < 0.6

### 2. Multi-class Classification

- One curve per class
- Micro-average: Overall performance
- Macro-average: Class-wise average
- Weighted average: Class-weighted performance

### 3. Average Precision

- Range: 0 to 1
- 0.5: Random classifier
- 1.0: Perfect classifier
- 0.7-0.8: Good classifier
- 0.8-0.9: Very good classifier
- 0.9+: Excellent classifier

## Best Practices

1. **Choose Appropriate Threshold**
   - Consider business costs
   - Balance precision and recall
   - Use domain knowledge
   - Validate with stakeholders

2. **Handle Class Imbalance**
   - Use appropriate sampling
   - Consider class weights
   - Apply cost-sensitive learning
   - Use balanced metrics

3. **Validate Results**
   - Use cross-validation
   - Check for overfitting
   - Compare with baseline
   - Consider multiple metrics

4. **Visualize Effectively**
   - Clear labels and title
   - Proper color scheme
   - Informative legend
   - Grid lines

## Common Mistakes to Avoid

1. **Ignoring Threshold Selection**
   - Using default threshold
   - Not considering costs
   - Missing business context
   - Overlooking trade-offs

2. **Poor Visualization**
   - Unclear labels
   - Wrong color scheme
   - Missing context
   - Incomplete information

3. **Misinterpretation**
   - Focusing on AUC alone
   - Ignoring class imbalance
   - Overlooking costs
   - Missing patterns

## Practical Example: Credit Risk Prediction

Let's analyze precision-recall curves for a credit risk prediction model:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

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

# Get prediction probabilities
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
average_precision = average_precision_score(y_test, y_pred_proba)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Credit Risk Prediction')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()
```

## Additional Resources

1. Scikit-learn documentation on precision-recall curves
2. Research papers on classification metrics
3. Online tutorials on model evaluation
4. Books on machine learning evaluation
