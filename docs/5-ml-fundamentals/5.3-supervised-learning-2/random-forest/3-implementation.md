# Implementing Random Forest 💻

Let's put theory into practice! We'll explore how to implement Random Forests using scikit-learn, from basic usage to advanced techniques.

## Basic Implementation 🚀

### Simple Classification Example
```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Create sample dataset
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))
```

## Real-World Example: Credit Risk Prediction 💳

```python
# Create realistic credit dataset
data = pd.DataFrame({
    'income': np.random.normal(50000, 20000, 1000),
    'age': np.random.normal(40, 10, 1000),
    'employment_length': np.random.normal(8, 4, 1000),
    'debt_ratio': np.random.uniform(0.1, 0.6, 1000),
    'credit_score': np.random.normal(700, 50, 1000)
})

# Create target variable
data['risk'] = (
    (data['debt_ratio'] > 0.4) & 
    (data['credit_score'] < 650)
).astype(int)

# Prepare features and target
X = data.drop('risk', axis=1)
y = data['risk']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model with best practices
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

# Train model
rf.fit(X_train, y_train)

# Print OOB score
print(f"Out-of-bag score: {rf.oob_score_:.3f}")

# Make predictions with probability
y_prob = rf.predict_proba(X_test)
risk_scores = y_prob[:, 1]  # Probability of high risk

# Create risk categories
risk_categories = pd.cut(
    risk_scores,
    bins=[0, 0.3, 0.6, 1],
    labels=['Low', 'Medium', 'High']
)

# Print distribution
print("\nRisk Distribution:")
print(risk_categories.value_counts())
```

## Feature Importance Analysis 📊

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, feature_names):
    """Plot feature importance with error bars"""
    importances = model.feature_importances_
    std = np.std([
        tree.feature_importances_ 
        for tree in model.estimators_
    ], axis=0)
    
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    
    plt.bar(range(X.shape[1]), 
            importances[indices],
            yerr=std[indices],
            align="center")
    
    plt.xticks(range(X.shape[1]), 
               [feature_names[i] for i in indices],
               rotation=45)
    plt.tight_layout()
    plt.show()

# Plot importance
plot_feature_importance(rf, X.columns)
```

## Hyperparameter Tuning 🎛️

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter space
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [None] + list(range(10, 50, 10)),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# Create random search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)

# Fit random search
random_search.fit(X_train, y_train)

# Print best parameters
print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)
```

## Advanced Techniques 🔧

### 1. Custom Scorer
```python
from sklearn.metrics import make_scorer, fbeta_score

# Create custom scorer that favors precision
beta = 0.5  # Weighs precision more than recall
f_half_scorer = make_scorer(
    fbeta_score, beta=beta
)

# Use in cross-validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    rf, X, y, 
    scoring=f_half_scorer,
    cv=5
)
print(f"F-{beta} scores:", scores)
```

### 2. Feature Selection
```python
from sklearn.feature_selection import SelectFromModel

# Select important features
selector = SelectFromModel(
    rf, prefit=True,
    threshold='median'  # Use median importance as threshold
)

# Transform data
X_selected = selector.transform(X)
print(f"Selected {X_selected.shape[1]} features")

# Get selected feature names
selected_features = X.columns[selector.get_support()].tolist()
print("Selected features:", selected_features)
```

### 3. Handling Imbalanced Data
```python
from imblearn.ensemble import BalancedRandomForestClassifier

# Create balanced random forest
brf = BalancedRandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train and evaluate
brf.fit(X_train, y_train)
y_pred_balanced = brf.predict(X_test)

print("\nBalanced Random Forest Results:")
print(classification_report(y_test, y_pred_balanced))
```

## Best Practices 🌟

### 1. Model Evaluation
```python
def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Comprehensive model evaluation"""
    # Training metrics
    train_pred = model.predict(X_train)
    train_prob = model.predict_proba(X_train)[:, 1]
    
    # Testing metrics
    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]
    
    # Print results
    print("Training Results:")
    print(classification_report(y_train, train_pred))
    print("\nTesting Results:")
    print(classification_report(y_test, test_pred))
    
    # Plot ROC curve
    from sklearn.metrics import plot_roc_curve
    plot_roc_curve(model, X_test, y_test)
    plt.show()
```

### 2. Feature Engineering
```python
def create_interaction_features(X):
    """Create interaction features"""
    X = X.copy()
    
    # Ratio features
    X['income_per_age'] = X['income'] / X['age']
    X['debt_per_income'] = X['debt_ratio'] * X['income']
    
    # Polynomial features
    X['credit_score_squared'] = X['credit_score'] ** 2
    
    return X

# Use in pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('feature_engineering', FunctionTransformer(create_interaction_features)),
    ('random_forest', RandomForestClassifier())
])
```

### 3. Model Persistence
```python
import joblib

# Save model
joblib.dump(rf, 'random_forest_model.joblib')

# Load model
loaded_rf = joblib.load('random_forest_model.joblib')
```

## Common Pitfalls and Solutions 🚧

1. **Memory Issues**
   ```python
   # Use smaller data types
   X = X.astype(np.float32)
   
   # Reduce number of trees
   rf = RandomForestClassifier(n_estimators=50)
   ```

2. **Long Training Time**
   ```python
   # Use fewer trees for initial experiments
   rf_quick = RandomForestClassifier(
       n_estimators=10,
       max_depth=5
   )
   
   # Use parallel processing
   rf.n_jobs = -1
   ```

3. **Overfitting**
   ```python
   # Increase min_samples_leaf
   rf = RandomForestClassifier(
       min_samples_leaf=5,
       max_depth=10
   )
   ```

## Next Steps 🚀

Ready to explore advanced techniques? Continue to [Advanced Topics](4-advanced.md)!
