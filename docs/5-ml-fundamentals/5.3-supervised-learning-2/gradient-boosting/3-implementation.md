# Implementing Gradient Boosting 💻

Let's explore how to implement Gradient Boosting using popular frameworks like XGBoost, LightGBM, and CatBoost!

## Basic Implementation with XGBoost 🚀

### Classification Example
```python
import numpy as np
import pandas as pd
import xgboost as xgb
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

# Create DMatrix (XGBoost's optimized data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'nthread': 4
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

# Make predictions
y_pred = model.predict(dtest)
y_pred_binary = (y_pred > 0.5).astype(int)

# Evaluate
print(classification_report(y_test, y_pred_binary))
```

## LightGBM Implementation 🌟

### Regression Example
```python
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

# Create regression dataset
X, y = make_regression(
    n_samples=1000,
    n_features=20,
    noise=0.1,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Set parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Train model
model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[train_data, test_data],
    callbacks=[lgb.early_stopping(10)]
)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

## CatBoost Implementation 🐱

### Handling Categorical Features
```python
from catboost import CatBoostClassifier, Pool

# Create dataset with categorical features
data = pd.DataFrame({
    'age': np.random.normal(40, 10, 1000),
    'income': np.random.normal(50000, 20000, 1000),
    'education': np.random.choice(['HS', 'BS', 'MS', 'PhD'], 1000),
    'occupation': np.random.choice(['Tech', 'Finance', 'Healthcare'], 1000)
})

# Create target
data['target'] = (
    (data['age'] > 35) & 
    (data['income'] > 45000) |
    (data['education'].isin(['MS', 'PhD']))
).astype(int)

# Split data
X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Specify categorical features
cat_features = ['education', 'occupation']

# Create CatBoost pools
train_pool = Pool(
    X_train, 
    y_train,
    cat_features=cat_features
)
test_pool = Pool(
    X_test,
    y_test,
    cat_features=cat_features
)

# Initialize and train model
model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    verbose=False
)
model.fit(train_pool, eval_set=test_pool)

# Make predictions
y_pred = model.predict(test_pool)

# Evaluate
print(classification_report(y_test, y_pred))
```

## Real-World Example: Customer Churn Prediction 🔄

```python
# Create realistic customer data
data = pd.DataFrame({
    'tenure': np.random.normal(30, 15, 1000),
    'monthly_charges': np.random.normal(70, 20, 1000),
    'total_charges': np.random.normal(2000, 800, 1000),
    'contract_type': np.random.choice(
        ['Month-to-month', 'One year', 'Two year'], 1000
    ),
    'payment_method': np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer'], 1000
    ),
    'internet_service': np.random.choice(
        ['DSL', 'Fiber optic', 'No'], 1000
    )
})

# Create target (churn probability)
data['churn'] = (
    (data['tenure'] < 12) & 
    (data['monthly_charges'] > 80) |
    (data['contract_type'] == 'Month-to-month')
).astype(int)

# Prepare features
cat_features = ['contract_type', 'payment_method', 'internet_service']
X = data.drop('churn', axis=1)
y = data['churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create CatBoost model
model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    verbose=False
)

# Train model
model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test)
)

# Feature importance analysis
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(importance)

# Make predictions with probability
y_prob = model.predict_proba(X_test)[:, 1]

# Create risk categories
risk_categories = pd.cut(
    y_prob,
    bins=[0, 0.3, 0.6, 1],
    labels=['Low', 'Medium', 'High']
)

print("\nRisk Distribution:")
print(risk_categories.value_counts())
```

## Best Practices 🌟

### 1. Data Preparation
```python
def prepare_data(df):
    """Prepare data for gradient boosting"""
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Encode categorical variables
    cat_columns = df.select_dtypes(['object']).columns
    df = pd.get_dummies(df, columns=cat_columns)
    
    # Scale numerical features (optional for tree-based models)
    num_columns = df.select_dtypes(['float64', 'int64']).columns
    scaler = StandardScaler()
    df[num_columns] = scaler.fit_transform(df[num_columns])
    
    return df
```

### 2. Parameter Tuning
```python
from sklearn.model_selection import RandomizedSearchCV

def tune_xgboost(X, y):
    """Tune XGBoost hyperparameters"""
    param_dist = {
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        random_state=42
    )
    
    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X, y)
    return random_search.best_params_
```

### 3. Model Evaluation
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
    plot_roc_curve(model, X_test, y_test)
    plt.show()
```

## Common Pitfalls and Solutions 🚧

1. **Memory Issues**
   ```python
   # Use LightGBM for large datasets
   model = lgb.LGBMClassifier(
       objective='binary',
       tree_learner='data'
   )
   ```

2. **Overfitting**
   ```python
   # Add regularization
   model = xgb.XGBClassifier(
       max_depth=3,
       min_child_weight=3,
       gamma=0.1,
       subsample=0.8,
       colsample_bytree=0.8
   )
   ```

3. **Slow Training**
   ```python
   # Use GPU acceleration
   model = CatBoostClassifier(
       task_type='GPU',
       devices='0:1'  # Use GPUs 0 and 1
   )
   ```

## Next Steps 🚀

Ready to explore advanced techniques? Continue to [Advanced Topics](4-advanced.md)!
