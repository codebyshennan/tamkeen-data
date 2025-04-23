# Implementing Random Forest

Let's learn how to implement Random Forests in Python using scikit-learn. We'll start with simple examples and gradually move to more complex applications.

## Basic Implementation

### Simple Classification Example

Let's start with a basic example that shows how to create and train a Random Forest classifier:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Create a sample dataset
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=1000,  # Number of samples
    n_features=20,   # Number of features
    n_informative=15,  # Number of informative features
    n_redundant=5,     # Number of redundant features
    random_state=42    # For reproducibility
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,    # 20% of data for testing
    random_state=42   # For reproducibility
)

# Create and train the Random Forest model
rf = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=10,      # Maximum depth of each tree
    random_state=42    # For reproducibility
)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
```

![Decision Tree vs Random Forest](assets/decision_tree_boundary.png)
*Figure 1: A single decision tree (left) makes simple, piecewise linear decisions, while a Random Forest (right) combines multiple trees to create more complex decision boundaries.*

## Real-World Example: Credit Risk Prediction

Let's create a more realistic example that shows how Random Forest can be used in a real-world scenario:

```python
# Create a realistic credit dataset
data = pd.DataFrame({
    'income': np.random.normal(50000, 20000, 1000),  # Annual income
    'age': np.random.normal(40, 10, 1000),           # Age in years
    'employment_length': np.random.normal(8, 4, 1000),  # Years employed
    'debt_ratio': np.random.uniform(0.1, 0.6, 1000),   # Debt to income ratio
    'credit_score': np.random.normal(700, 50, 1000)    # Credit score
})

# Create target variable (high risk = 1, low risk = 0)
data['risk'] = (
    (data['debt_ratio'] > 0.4) & 
    (data['credit_score'] < 650)
).astype(int)

# Prepare features and target
X = data.drop('risk', axis=1)
y = data['risk']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# Create model with best practices
rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=None,        # Let trees grow fully
    min_samples_split=5,   # Minimum samples to split
    min_samples_leaf=2,    # Minimum samples in leaf
    max_features='sqrt',   # Number of features to consider
    bootstrap=True,        # Use bootstrap sampling
    oob_score=True,        # Calculate out-of-bag score
    random_state=42,       # For reproducibility
    n_jobs=-1             # Use all CPU cores
)

# Train the model
rf.fit(X_train, y_train)

# Print out-of-bag score
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

## Feature Importance Analysis

Understanding which features are most important for making predictions:

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance with error bars.
    
    Parameters:
    model: Trained Random Forest model
    feature_names: Names of the features
    """
    # Get feature importances
    importances = model.feature_importances_
    
    # Calculate standard deviation of importances
    std = np.std([
        tree.feature_importances_ 
        for tree in model.estimators_
    ], axis=0)
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    
    # Plot bars with error bars
    plt.bar(range(X.shape[1]), 
            importances[indices],
            yerr=std[indices],
            align="center")
    
    # Add feature names
    plt.xticks(range(X.shape[1]), 
               [feature_names[i] for i in indices],
               rotation=45)
    plt.tight_layout()
    plt.show()

# Plot importance
plot_feature_importance(rf, X.columns)
```

![Feature Importance](assets/feature_importance.png)
*Figure 2: Feature importance shows which features contribute most to the model's predictions.*

## Hyperparameter Tuning

Finding the best combination of parameters for your model:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter space to search
param_dist = {
    'n_estimators': randint(100, 500),  # Number of trees
    'max_depth': [None] + list(range(10, 50, 10)),  # Tree depth
    'min_samples_split': randint(2, 20),  # Min samples to split
    'min_samples_leaf': randint(1, 10),   # Min samples in leaf
    'max_features': ['sqrt', 'log2', None],  # Features to consider
    'bootstrap': [True, False]  # Whether to use bootstrap
}

# Create random search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=100,  # Number of parameter settings to try
    cv=5,        # 5-fold cross-validation
    scoring='roc_auc',  # Metric to optimize
    n_jobs=-1,   # Use all CPU cores
    random_state=42
)

# Fit random search
random_search.fit(X_train, y_train)

# Print best parameters
print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)
```

![Bias-Variance Tradeoff](assets/bias_variance.png)
*Figure 3: The bias-variance tradeoff in Random Forests - how model complexity affects predictions.*

## Advanced Techniques

### 1. Custom Scorer

Creating a custom scoring metric that favors precision over recall:

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

Selecting only the most important features:

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

Using a balanced version of Random Forest for imbalanced datasets:

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

![Ensemble Prediction](assets/ensemble_prediction.png)
*Figure 4: How individual tree predictions combine to form the final ensemble prediction.*

## Best Practices

### 1. Model Evaluation

Comprehensive evaluation of model performance:

```python
def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Comprehensive model evaluation.
    
    Parameters:
    model: Trained model
    X_train, X_test: Training and test features
    y_train, y_test: Training and test targets
    """
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

Creating new features to improve model performance:

```python
def create_interaction_features(X):
    """
    Create interaction features.
    
    Parameters:
    X: Original features
    
    Returns:
    X: Features with new interaction terms
    """
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

Saving and loading trained models:

```python
import joblib

# Save model
joblib.dump(rf, 'random_forest_model.joblib')

# Load model
loaded_rf = joblib.load('random_forest_model.joblib')
```

## Common Pitfalls and Solutions

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

## Next Steps

Ready to explore advanced techniques? Continue to [Advanced Topics](4-advanced.md)!
