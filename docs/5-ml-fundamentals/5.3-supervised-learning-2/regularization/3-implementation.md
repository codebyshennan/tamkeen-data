# Implementing Regularization

Think of implementing regularization like learning to ride a bicycle - we'll start with the basics and gradually build up to more advanced techniques. Let's make this journey as smooth as possible!

## Basic Implementation

### Simple Example with Ridge Regression

Let's start with a basic example that shows how to implement Ridge Regression, one of the most common regularization techniques.

```python
# Import necessary libraries
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Create sample data
# Think of this as creating a small dataset to practice with
np.random.seed(42)  # This ensures we get the same random numbers each time
X = np.random.randn(100, 20)  # 100 samples with 20 features
y = np.random.randn(100)      # 100 target values

# Split and scale data
# This is like dividing your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features
# This is like converting different currencies to a common standard
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train ridge model
# Think of alpha as how strict the regularization is
ridge = Ridge(alpha=0.1)  # alpha is the regularization strength
ridge.fit(X_train_scaled, y_train)

# Make predictions
# This is like using your trained model to make new predictions
y_pred = ridge.predict(X_test_scaled)
```

## Real-World Example: House Price Prediction

Let's look at a more practical example that you might encounter in the real world - predicting house prices.

```python
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create sample dataset
# This simulates a real-world housing dataset
np.random.seed(42)
n_samples = 1000

# Create features that might affect house prices
data = pd.DataFrame({
    'size': np.random.normal(2000, 500, n_samples),  # House size in square feet
    'bedrooms': np.random.randint(1, 6, n_samples),  # Number of bedrooms
    'age': np.random.randint(0, 50, n_samples),      # Age of the house
    'location_score': np.random.uniform(0, 10, n_samples),  # Location quality
    'noise_feature1': np.random.normal(0, 1, n_samples),    # Random noise
    'noise_feature2': np.random.normal(0, 1, n_samples)     # Random noise
})

# Create target with noise
# This simulates how house prices are determined
data['price'] = (
    200 * data['size'] 
    + 50000 * data['bedrooms']
    - 1000 * data['age']
    + 20000 * data['location_score']
    + np.random.normal(0, 50000, n_samples)  # Add some randomness
)

# Prepare data
X = data.drop('price', axis=1)  # Features
y = data['price']               # Target

# Scale features
# This is crucial for regularization to work properly
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
# This helps us evaluate how well our model generalizes
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Compare different regularization methods
# Let's see how different types of regularization perform
from sklearn.linear_model import Lasso, ElasticNet

models = {
    'Lasso': Lasso(alpha=0.1),           # L1 regularization
    'Ridge': Ridge(alpha=0.1),           # L2 regularization
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)  # Combined L1 and L2
}

results = {}
for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Store results
    results[name] = {
        'R2': r2_score(y_test, y_pred),  # How well the model fits
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),  # Average error
        'Coefficients': model.coef_  # How important each feature is
    }

# Plot coefficients comparison
# This helps us visualize how different regularization methods affect feature importance
plt.figure(figsize=(12, 6))
x = np.arange(len(X.columns))
width = 0.25

for i, (name, result) in enumerate(results.items()):
    plt.bar(x + i*width, result['Coefficients'], 
            width, label=name)

plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Comparison of Regularization Methods')
plt.xticks(x + width, X.columns, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Print performance metrics
# This helps us compare how well each method performs
for name, result in results.items():
    print(f"\n{name}:")
    print(f"R Score: {result['R2']:.3f}")  # Higher is better
    print(f"RMSE: ${result['RMSE']:,.2f}")  # Lower is better
```

## Hyperparameter Tuning

Finding the right regularization strength (alpha) is like finding the right amount of seasoning for a dish - too little and it's bland, too much and it's overwhelming.

```python
from sklearn.model_selection import GridSearchCV

# Setup parameter grid
# We'll try different values of alpha to find the best one
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}

# Perform grid search
# This is like trying different amounts of seasoning to find the perfect taste
grid_search = GridSearchCV(
    Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error'
)
grid_search.fit(X_train, y_train)

print("Best alpha:", grid_search.best_params_['alpha'])
```

## Feature Selection with Lasso

Lasso regularization is particularly good at feature selection - it's like having a strict teacher who helps you focus on the most important subjects.

```python
def select_features_lasso(X, y, alpha=0.1):
    """Select features using Lasso regularization"""
    # Train Lasso model
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    
    # Get selected features
    selected_features = X.columns[lasso.coef_ != 0]
    
    # Print results
    print("Selected features:", len(selected_features))
    for feature, coef in zip(X.columns, lasso.coef_):
        if coef != 0:
            print(f"{feature}: {coef:.4f}")
    
    return selected_features

# Use function
selected = select_features_lasso(
    pd.DataFrame(X_train, columns=X.columns), 
    y_train
)
```

## Cross-Validation Implementation

Cross-validation is like taking multiple tests to ensure you really understand the material, not just memorizing the answers.

```python
from sklearn.model_selection import cross_val_score

def compare_alphas(X, y, alphas=[0.001, 0.01, 0.1, 1, 10]):
    """Compare different regularization strengths"""
    results = []
    
    for alpha in alphas:
        # Create and evaluate models
        ridge_scores = cross_val_score(
            Ridge(alpha=alpha), X, y, 
            cv=5, scoring='neg_mean_squared_error'
        )
        lasso_scores = cross_val_score(
            Lasso(alpha=alpha), X, y,
            cv=5, scoring='neg_mean_squared_error'
        )
        elastic_scores = cross_val_score(
            ElasticNet(alpha=alpha, l1_ratio=0.5), X, y,
            cv=5, scoring='neg_mean_squared_error'
        )
        
        # Store results
        results.append({
            'alpha': alpha,
            'ridge_rmse': np.sqrt(-ridge_scores.mean()),
            'lasso_rmse': np.sqrt(-lasso_scores.mean()),
            'elastic_rmse': np.sqrt(-elastic_scores.mean())
        })
    
    return pd.DataFrame(results)

# Compare alphas
results_df = compare_alphas(X_scaled, y)
print(results_df)
```

## Model Evaluation Functions

Evaluating your model is like checking your work after solving a problem - it helps you understand how well you're doing.

```python
def evaluate_regularized_model(model, X_train, X_test, y_train, y_test):
    """Comprehensive evaluation of regularized model"""
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    results = {
        'train_r2': r2_score(y_train, y_train_pred),  # How well it fits training data
        'test_r2': r2_score(y_test, y_test_pred),     # How well it generalizes
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),  # Training error
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),     # Testing error
        'n_nonzero_coef': np.sum(model.coef_ != 0)    # How many features it uses
    }
    
    # Print results
    print("Training R²:", results['train_r2'])
    print("Testing R²:", results['test_r2'])
    print("Training RMSE:", results['train_rmse'])
    print("Testing RMSE:", results['test_rmse'])
    print("Non-zero coefficients:", results['n_nonzero_coef'])
    
    return results
```

## Best Practices

### 1. Feature Scaling

Always scale your features before applying regularization. This is like converting different currencies to a common standard before comparing them.

```python
# Always scale features before regularization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 2. Cross-Validation

Use cross-validation to find the best regularization strength. This is like taking multiple tests to ensure you really understand the material.

### 3. Feature Selection

Consider using Lasso for feature selection when you have many features. This is like having a strict teacher who helps you focus on the most important subjects.

### 4. Model Comparison

Compare different regularization methods to find the best one for your specific problem. This is like trying different approaches to solve a problem.

### 5. Regularization Path

Plot the regularization path to understand how different features are affected by regularization. This is like seeing how different ingredients affect the taste of a dish.

## Common Mistakes to Avoid

1. Not scaling features before regularization
2. Using the same regularization strength for all features
3. Not validating the regularization effect
4. Ignoring feature selection when appropriate
5. Not comparing different regularization methods

## Next Steps

Now that you understand how to implement regularization, let's move on to [Advanced Topics](4-advanced.md) to explore more sophisticated techniques!

## Additional Resources

- [Scikit-learn Regularization Documentation](https://scikit-learn.org/stable/modules/linear_model.html)
- [Regularization in Machine Learning](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a)
- [Understanding L1 and L2 Regularization](https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/)
