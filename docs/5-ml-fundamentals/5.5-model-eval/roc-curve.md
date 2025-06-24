# ROC Curve

## Introduction

The Receiver Operating Characteristic (ROC) curve is a fundamental tool in machine learning for evaluating classification models. It provides a visual representation of a model's performance across different classification thresholds.

## What is an ROC Curve?

An ROC curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings. It helps visualize the trade-off between sensitivity and specificity.

![ROC Curve](assets/roc_curve.png)

**Key Components:**
- **True Positive Rate (TPR)** = Sensitivity = Recall = TP / (TP + FN)
  - "Of all actual positive cases, how many did we correctly identify?"
- **False Positive Rate (FPR)** = 1 - Specificity = FP / (FP + TN)  
  - "Of all actual negative cases, how many did we incorrectly identify as positive?"

**Understanding the Curve:**
- **Perfect Classifier**: Curve goes straight up to (0,1) then across to (1,1) - AUC = 1.0
- **Random Classifier**: Diagonal line from (0,0) to (1,1) - AUC = 0.5
- **Good Classifier**: Curve bows toward the upper-left corner - AUC > 0.7
- **Poor Classifier**: Curve below diagonal line - AUC < 0.5

### Real-World Example: Email Spam Detection

Imagine an email spam filter:
- **High TPR, Low FPR**: Catches most spam (good!) without blocking legitimate emails (good!)
- **High TPR, High FPR**: Catches spam but also blocks important emails (bad!)
- **Low TPR, Low FPR**: Doesn't block legitimate emails but lets spam through (bad!)
- **Low TPR, High FPR**: Worst case - misses spam AND blocks legitimate emails!

## Types of ROC Curves

### 1. Binary Classification

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
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

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

### 2. Multi-class Classification

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
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

# Calculate ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curves')
plt.legend(loc="lower right")
plt.show()
```

## Interpreting ROC Curves

### 1. Binary Classification

- Area Under Curve (AUC): Overall model performance
- Diagonal line: Random classifier
- Upper left corner: Ideal classifier
- Curve shape: Model's discriminative ability

### 2. Multi-class Classification

- One curve per class
- Micro-average: Overall performance
- Macro-average: Class-wise average
- Weighted average: Class-weighted performance

### 3. AUC Score

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

Let's analyze ROC curves for a credit risk prediction model:

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

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Credit Risk Prediction')
plt.legend(loc="lower right")
plt.show()
```

## Additional Resources

1. Scikit-learn documentation on ROC curves
2. Research papers on classification metrics
3. Online tutorials on model evaluation
4. Books on machine learning evaluation
