# Gradient Boosting

Imagine learning from your mistakes - each time you make a prediction, you focus more on the examples you got wrong. That's exactly how Gradient Boosting works! It builds models sequentially, with each new model trying to correct the errors of previous ones. Let's dive into this powerful technique! 🚀

## Understanding Gradient Boosting 🎯

Gradient Boosting works by:
1. Building a simple initial model
2. Calculating errors (residuals)
3. Building new models to predict these errors
4. Combining all models with appropriate weights

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression

# Create sample dataset
X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)

# Visualize boosting stages
stages = [1, 3, 5, 10]
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for idx, n_estimators in enumerate(stages):
    gb = GradientBoostingRegressor(n_estimators=n_estimators, 
                                  learning_rate=0.1,
                                  random_state=42)
    gb.fit(X, y)
    
    # Plot predictions
    X_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = gb.predict(X_test)
    
    axes[idx].scatter(X, y, alpha=0.1)
    axes[idx].plot(X_test, y_pred, color='red', linewidth=2)
    axes[idx].set_title(f'n_estimators={n_estimators}')

plt.tight_layout()
plt.show()
```

## Popular Implementations 🛠️

### 1. Scikit-learn's GradientBoosting
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create dataset
X, y = make_classification(n_samples=1000, n_features=20,
                         n_informative=15, n_redundant=5,
                         random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)

# Evaluate
print(f"Training score: {gb.score(X_train, y_train):.3f}")
print(f"Testing score: {gb.score(X_test, y_test):.3f}")
```

### 2. XGBoost
```python
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Create DMatrix (XGBoost's optimized data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

# Train model
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=10,
    verbose_eval=False
)

# Plot feature importance
xgb.plot_importance(model)
plt.show()
```

### 3. LightGBM
```python
import lightgbm as lgb

# Create dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Set parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.1
}

# Train model
model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[train_data, test_data],
    early_stopping_rounds=10,
    verbose_eval=False
)

# Plot feature importance
lgb.plot_importance(model)
plt.show()
```

## Real-World Example: House Price Prediction 🏠

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Create sample housing dataset
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'size': np.random.normal(2000, 500, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'age': np.random.randint(0, 50, n_samples),
    'location_score': np.random.uniform(0, 10, n_samples)
})

# Create target variable with some noise
data['price'] = (
    200 * data['size'] 
    + 50000 * data['bedrooms']
    - 1000 * data['age']
    + 20000 * data['location_score']
    + np.random.normal(0, 50000, n_samples)
)

# Prepare data
X = data.drop('price', axis=1)
y = data['price']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train XGBoost model
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'max_depth': 5,
    'eta': 0.1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=10,
    verbose_eval=False
)

# Make predictions
predictions = model.predict(dtest)

# Evaluate
print(f"R² Score: {r2_score(y_test, predictions):.3f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, predictions)):,.2f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.tight_layout()
plt.show()
```

## Advanced Techniques 🔧

### 1. Learning Rate Scheduling
```python
def learning_rate_scheduler(iteration):
    """Reduce learning rate over time"""
    initial_rate = 0.1
    decay = 0.995
    return initial_rate * (decay ** iteration)

# Use in XGBoost
params['callbacks'] = [xgb.callback.reset_learning_rate(learning_rate_scheduler)]
```

### 2. Feature Interactions
```python
# Create interaction features
X['size_per_bedroom'] = X['size'] / X['bedrooms']
X['age_value_ratio'] = X['age'] / X['price']
```

### 3. Custom Loss Functions
```python
def custom_objective(predt, dtrain):
    """Custom objective function for XGBoost"""
    y = dtrain.get_label()
    grad = 2 * (predt - y)
    hess = 2 * np.ones_like(predt)
    return grad, hess

# Use custom objective
params['objective'] = custom_objective
```

## Best Practices 🌟

### 1. Prevent Overfitting
```python
params = {
    'max_depth': 3,
    'min_child_weight': 1,
    'gamma': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}
```

### 2. Handle Missing Values
```python
# XGBoost handles missing values automatically
# For other implementations:
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X_clean = imputer.fit_transform(X)
```

### 3. Feature Selection
```python
# Use feature importance for selection
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importance()
}).sort_values('importance', ascending=False)

# Select top features
top_features = importance.head(10)['feature'].tolist()
```

## Common Pitfalls and Solutions 🚧

1. **Overfitting**
   - Reduce model complexity
   - Use early stopping
   - Increase regularization

2. **Long Training Time**
   - Use fewer trees
   - Sample data for tuning
   - Try LightGBM for speed

3. **Memory Issues**
   - Reduce max_depth
   - Use data sampling
   - Try streaming/chunking

## Next Steps

Now that you understand Gradient Boosting, let's explore [Neural Networks](./neural-networks.md) to learn about deep learning!
