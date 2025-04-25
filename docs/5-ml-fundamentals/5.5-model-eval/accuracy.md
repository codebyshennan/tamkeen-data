# Accuracy

## Introduction

Accuracy is one of the most fundamental metrics in machine learning, measuring the proportion of correct predictions made by a model. While simple to understand and calculate, it's important to use accuracy appropriately and understand its limitations.

## What is Accuracy?

Accuracy is the ratio of correct predictions to total predictions:

\[
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
\]

## Types of Accuracy

### 1. Binary Classification

```python
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

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
```

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

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Calculate baseline accuracy
baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())
print(f"Baseline Accuracy: {baseline_accuracy:.3f}")
```

## Additional Resources

1. Scikit-learn documentation on accuracy metrics
2. Research papers on classification metrics
3. Online tutorials on model evaluation
4. Books on machine learning evaluation
