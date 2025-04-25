# ROC and AUC

## Introduction

The Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) are powerful tools for evaluating binary classification models. They provide a comprehensive view of model performance across different classification thresholds.

## What is ROC?

The ROC curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings:

\[
\text{TPR} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
\]

\[
\text{FPR} = \frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}}
\]

## What is AUC?

The Area Under the ROC Curve (AUC) measures the model's ability to distinguish between classes:

- AUC = 1.0: Perfect classification
- AUC = 0.5: Random guessing
- AUC < 0.5: Worse than random

## Implementation

### 1. Basic ROC Curve

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

# Get probability predictions
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

### 2. Multiple Models Comparison

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

plt.figure(figsize=(8, 6))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Multiple Models')
plt.legend(loc="lower right")
plt.show()
```

## Interpreting ROC and AUC

### 1. ROC Curve Interpretation

- Upper left corner: Ideal point
- Diagonal line: Random guessing
- Curve shape: Model performance

### 2. AUC Interpretation

- AUC > 0.9: Excellent
- 0.7 < AUC < 0.9: Good
- 0.5 < AUC < 0.7: Fair
- AUC = 0.5: Random

### 3. Threshold Selection

- Operating point: Business requirements
- Cost-benefit analysis
- Class imbalance

## Best Practices

1. **Data Preparation**
   - Handle missing values
   - Scale features
   - Balance classes

2. **Model Selection**
   - Compare multiple models
   - Use cross-validation
   - Consider complexity

3. **Evaluation**
   - Use multiple metrics
   - Consider costs
   - Validate results

4. **Threshold Selection**
   - Business requirements
   - Cost-benefit analysis
   - Class imbalance

## Common Mistakes to Avoid

1. **Data Issues**
   - Not handling missing values
   - Ignoring class imbalance
   - Skipping preprocessing

2. **Model Selection**
   - Overfitting
   - Underfitting
   - Ignoring complexity

3. **Evaluation**
   - Relying solely on AUC
   - Ignoring costs
   - Not validating

## Practical Example: Credit Risk Prediction

Let's analyze ROC and AUC for a credit risk prediction model:

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

# Get probability predictions
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

1. Scikit-learn documentation on ROC and AUC
2. Research papers on classification metrics
3. Online tutorials on model evaluation
4. Books on machine learning evaluation
