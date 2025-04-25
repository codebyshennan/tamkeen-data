# Validation Curves

## Introduction

Validation curves are essential tools in machine learning for understanding how a model's performance changes with different hyperparameter values. They help us find the optimal hyperparameter settings and diagnose issues like overfitting and underfitting.

## What are Validation Curves?

Validation curves plot the model's performance (typically error or accuracy) against different values of a hyperparameter. They show:

1. Training score
2. Validation score
3. The relationship between them

## Types of Validation Curves

### 1. Model Complexity

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)

# Calculate validation curves
param_range = np.arange(1, 11)
train_scores, val_scores = validation_curve(
    DecisionTreeClassifier(random_state=42), X, y,
    param_name="max_depth", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=-1)

# Calculate mean and std
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot validation curves
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, label='Training score')
plt.plot(param_range, val_mean, label='Cross-validation score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Max Depth')
plt.ylabel('Score')
plt.title('Validation Curves (Model Complexity)')
plt.legend(loc='best')
plt.grid(True)
plt.show()
```

### 2. Regularization Strength

```python
from sklearn.linear_model import LogisticRegression

# Calculate validation curves
param_range = np.logspace(-4, 4, 9)
train_scores, val_scores = validation_curve(
    LogisticRegression(random_state=42), X, y,
    param_name="C", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=-1)

# Calculate mean and std
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot validation curves
plt.figure(figsize=(10, 6))
plt.semilogx(param_range, train_mean, label='Training score')
plt.semilogx(param_range, val_mean, label='Cross-validation score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('C (Inverse Regularization Strength)')
plt.ylabel('Score')
plt.title('Validation Curves (Regularization)')
plt.legend(loc='best')
plt.grid(True)
plt.show()
```

### 3. Learning Rate

```python
from sklearn.ensemble import GradientBoostingClassifier

# Calculate validation curves
param_range = np.logspace(-3, 0, 10)
train_scores, val_scores = validation_curve(
    GradientBoostingClassifier(random_state=42), X, y,
    param_name="learning_rate", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=-1)

# Calculate mean and std
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot validation curves
plt.figure(figsize=(10, 6))
plt.semilogx(param_range, train_mean, label='Training score')
plt.semilogx(param_range, val_mean, label='Cross-validation score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Learning Rate')
plt.ylabel('Score')
plt.title('Validation Curves (Learning Rate)')
plt.legend(loc='best')
plt.grid(True)
plt.show()
```

## Interpreting Validation Curves

### 1. Overfitting

- Training score increases
- Validation score decreases
- Large gap between curves
- Need more regularization

### 2. Underfitting

- Both scores are low
- Small gap between curves
- Need more complexity
- More features might help

### 3. Good Fit

- Both scores are high
- Small gap between curves
- Optimal parameter found
- Model is well-tuned

## Best Practices

1. **Choose Appropriate Range**
   - Wide enough to see trends
   - Fine enough for precision
   - Log scale when needed

2. **Use Cross-Validation**
   - Multiple folds
   - Stratified sampling
   - Appropriate metrics

3. **Plot Confidence Intervals**
   - Show standard deviation
   - Multiple runs
   - Clear visualization

4. **Consider Multiple Parameters**
   - Grid search
   - Random search
   - Bayesian optimization

## Common Mistakes to Avoid

1. **Insufficient Range**
   - Too narrow
   - Missing optimal point
   - Wrong conclusions

2. **Poor Cross-Validation**
   - Not enough folds
   - Data leakage
   - Inappropriate metrics

3. **Misinterpretation**
   - Ignoring variance
   - Overlooking trends
   - Wrong conclusions

## Practical Example: Credit Risk Prediction

Let's analyze validation curves for a credit risk prediction model:

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

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Calculate validation curves
param_range = np.arange(1, 21)
train_scores, val_scores = validation_curve(
    pipeline, X, y,
    param_name="classifier__max_depth", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=-1)

# Calculate mean and std
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot validation curves
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, label='Training score')
plt.plot(param_range, val_mean, label='Cross-validation score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Validation Curves for Credit Risk Prediction')
plt.legend(loc='best')
plt.grid(True)
plt.show()
```

## Additional Resources

1. Scikit-learn documentation on validation curves
2. Research papers on hyperparameter tuning
3. Online tutorials on model evaluation
4. Books on machine learning optimization
