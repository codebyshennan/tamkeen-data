# Random Forest

Imagine having a committee of experts making decisions instead of just one person. That's exactly how Random Forests work - they combine multiple decision trees to make better predictions! Let's learn how to harness this powerful ensemble method. 🌳🌲🌳

## Understanding Random Forest 🎯

Random Forest combines two key ideas:
1. **Bagging (Bootstrap Aggregating)**: Training each tree on a random subset of data
2. **Random Feature Selection**: Each tree considers a random subset of features at each split

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Create sample dataset
X, y = make_classification(n_samples=1000, n_features=20,
                         n_informative=15, n_redundant=5,
                         random_state=42)

# Create and train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Plot feature importances
def plot_feature_importance(model, title="Feature Importance"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), [f"Feature {i}" for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()

plot_feature_importance(rf)
```

## How Random Forest Works 🔄

### 1. Bootstrap Sampling
```python
def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[idxs], y[idxs]

# Demonstrate bootstrap sampling
X_boot, y_boot = bootstrap_sample(X, y)
print(f"Original size: {len(y)}, Bootstrap size: {len(y_boot)}")
print(f"Unique samples: {len(np.unique(y_boot))}")
```

### 2. Random Feature Selection
```python
from sklearn.tree import DecisionTreeClassifier

# Create trees with different feature subsets
n_features = X.shape[1]
max_features = int(np.sqrt(n_features))  # Common rule of thumb

tree = DecisionTreeClassifier(max_features=max_features)
tree.fit(X, y)
```

### 3. Ensemble Prediction
```python
# Create ensemble of trees
trees = []
n_trees = 5

for _ in range(n_trees):
    X_boot, y_boot = bootstrap_sample(X, y)
    tree = DecisionTreeClassifier(max_features=max_features)
    tree.fit(X_boot, y_boot)
    trees.append(tree)

# Make predictions
def ensemble_predict(trees, X):
    predictions = np.array([tree.predict(X) for tree in trees])
    return np.mean(predictions, axis=0)  # For regression
    # Or for classification:
    # return np.apply_along_axis(lambda x: np.bincount(x).argmax(), 
    #                           axis=0, arr=predictions)
```

## Real-World Example: Credit Risk Prediction 💳

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Sample credit data
data = pd.DataFrame({
    'income': np.random.normal(50000, 20000, 1000),
    'age': np.random.normal(40, 10, 1000),
    'loan_amount': np.random.normal(200000, 100000, 1000),
    'employment_length': np.random.normal(8, 4, 1000)
})
data['default'] = (data['loan_amount'] > data['income'] * 4).astype(int)

# Split data
X = data.drop('default', axis=1)
y = data['default']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d',
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
plt.title('Confusion Matrix')
plt.show()
```

## Out-of-Bag (OOB) Score 📊

One unique feature of Random Forest is the ability to use out-of-bag samples for validation:

```python
rf_oob = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=42
)
rf_oob.fit(X, y)

print(f"OOB Score: {rf_oob.oob_score_:.3f}")
```

## Feature Importance Analysis 📈

```python
def plot_feature_importance_with_std(rf, feature_names):
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ 
                  for tree in rf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances with Standard Deviation")
    plt.bar(range(X.shape[1]), importances[indices],
            yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), 
               [feature_names[i] for i in indices],
               rotation=45)
    plt.tight_layout()
    plt.show()

# Plot feature importance with error bars
plot_feature_importance_with_std(rf, X.columns)
```

## Hyperparameter Tuning 🎛️

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
```

## Best Practices 🌟

### 1. Feature Engineering
```python
# Create interaction features
X['income_per_age'] = X['income'] / X['age']
X['loan_to_income'] = X['loan_amount'] / X['income']
```

### 2. Handling Imbalanced Data
```python
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y),
    y=y
)
class_weight_dict = dict(zip(np.unique(y), class_weights))

# Train with class weights
rf_balanced = RandomForestClassifier(
    class_weight=class_weight_dict,
    random_state=42
)
```

### 3. Feature Selection
```python
from sklearn.feature_selection import SelectFromModel

# Select important features
selector = SelectFromModel(rf, prefit=True)
X_selected = selector.transform(X)
print(f"Selected {X_selected.shape[1]} features")
```

## Common Pitfalls and Solutions 🚧

1. **Overfitting**
   - Reduce max_depth
   - Increase min_samples_split
   - Use fewer trees

2. **Underfitting**
   - Increase max_depth
   - Use more trees
   - Consider feature engineering

3. **Computational Issues**
   - Use fewer trees
   - Limit max_depth
   - Sample data for initial tuning

## Next Steps

Now that you understand Random Forests, let's explore [Gradient Boosting](./gradient-boosting.md) to learn about another powerful ensemble method!
