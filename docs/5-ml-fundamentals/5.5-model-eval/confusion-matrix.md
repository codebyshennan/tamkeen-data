# Confusion Matrix

## Introduction

A confusion matrix is a fundamental tool in machine learning for evaluating classification models. It provides a detailed breakdown of model predictions versus actual values, helping to understand model performance across different classes.

## What is a Confusion Matrix?

A confusion matrix is a table that describes the performance of a classification model by comparing predicted values with actual values. It shows:

- True Positives (TP): Correctly predicted positive cases
- True Negatives (TN): Correctly predicted negative cases
- False Positives (FP): Incorrectly predicted positive cases
- False Negatives (FN): Incorrectly predicted negative cases

## Types of Confusion Matrices

### 1. Binary Classification

```python
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
```

### 2. Multi-class Classification

```python
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
```

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
```

## Additional Resources

1. Scikit-learn documentation on confusion matrices
2. Research papers on classification metrics
3. Online tutorials on model evaluation
4. Books on machine learning evaluation
