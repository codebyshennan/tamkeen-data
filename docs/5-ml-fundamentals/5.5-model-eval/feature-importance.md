# Feature Importance

## Introduction

Feature importance is a crucial concept in machine learning that helps us understand which features contribute most to our model's predictions. This understanding is essential for model interpretability, feature selection, and domain knowledge validation.

## What is Feature Importance?

Feature importance measures how much each feature contributes to the model's predictions. It helps us:

1. Identify the most influential features
2. Remove irrelevant features
3. Understand model behavior
4. Validate domain knowledge

## Types of Feature Importance

### 1. Tree-Based Methods

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Create sample dataset
X, y = make_classification(n_samples=1000, n_features=10, 
                         n_informative=5, n_redundant=2,
                         random_state=42)

# Train random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [f'Feature {i}' for i in indices], rotation=45)
plt.tight_layout()
plt.show()
```

### 2. Permutation Importance

```python
from sklearn.inspection import permutation_importance

# Calculate permutation importance
result = permutation_importance(rf, X, y, n_repeats=10, random_state=42)

# Plot results
plt.figure(figsize=(10, 6))
plt.title('Permutation Importances')
plt.boxplot(result.importances.T, labels=[f'Feature {i}' for i in range(X.shape[1])])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 3. SHAP Values

```python
import shap

# Calculate SHAP values
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

# Plot summary
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, plot_type="bar")
plt.tight_layout()
plt.show()
```

## Best Practices

1. **Use Multiple Methods**
   - Combine different importance measures
   - Cross-validate results
   - Consider domain knowledge

2. **Handle Correlated Features**
   - Group correlated features
   - Use appropriate methods
   - Consider feature interactions

3. **Validate Results**
   - Use cross-validation
   - Check stability
   - Compare with domain knowledge

4. **Visualize Effectively**
   - Use appropriate plots
   - Show confidence intervals
   - Include feature names

## Common Mistakes to Avoid

1. **Ignoring Feature Correlations**
   - Not considering interactions
   - Missing important relationships
   - Overlooking multicollinearity

2. **Overlooking Scale**
   - Not normalizing features
   - Comparing different scales
   - Misinterpreting results

3. **Poor Visualization**
   - Unclear plots
   - Missing context
   - Inappropriate scales

## Practical Example: Credit Risk Prediction

Let's analyze feature importance in a credit risk prediction task:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit pipeline
pipeline.fit(X, y)

# Get feature importances
importances = pipeline.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title('Feature Importances in Credit Risk Prediction')
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45)
plt.tight_layout()
plt.show()

# Calculate and plot SHAP values
explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
shap_values = explainer.shap_values(X)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X)
plt.tight_layout()
plt.show()
```

## Additional Resources

1. Scikit-learn documentation on feature importance
2. SHAP documentation and examples
3. Research papers on feature selection
4. Online tutorials on model interpretability
