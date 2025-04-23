# Hyperparameter Tuning

## What are Hyperparameters? 🤔

Think of hyperparameters as the "settings" or "knobs" of your machine learning model. Just like how you adjust the temperature and cooking time when baking a cake, hyperparameters are the settings you can adjust to make your model perform better.

### Why Hyperparameter Tuning Matters 🌟

Imagine you're tuning a musical instrument. You need to adjust various settings (like string tension, bridge position, etc.) to get the perfect sound. Similarly, hyperparameter tuning helps us find the best settings for our model to achieve optimal performance.

## Real-World Analogies 📚

### The Car Engine Tuning Analogy

Think of hyperparameter tuning like tuning a car engine:

- The engine is your model
- The tuning parameters (spark timing, fuel mixture, etc.) are your hyperparameters
- The performance metrics (horsepower, fuel efficiency) are your model's metrics
- The tuning process is like finding the perfect balance of settings

### The Recipe Optimization Analogy

Hyperparameter tuning is like perfecting a recipe:

- The recipe is your model
- The ingredients and cooking times are your hyperparameters
- The taste test is your model evaluation
- The optimization process is like trying different combinations to get the perfect dish

## Types of Hyperparameter Tuning 🎯

### 1. Grid Search

This is like trying every possible combination of settings systematically.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Create sample dataset
X, y = make_classification(n_samples=1000, n_features=20, 
                         n_informative=15, n_redundant=5,
                         random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search
grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X, y)

# Visualize results
def plot_grid_search_results(grid_search):
    results = grid_search.cv_results_
    scores = results['mean_test_score'].reshape(len(param_grid['n_estimators']), 
                                              len(param_grid['max_depth']))
    
    plt.figure(figsize=(10, 8))
    plt.imshow(scores, cmap='viridis')
    plt.colorbar(label='Mean CV Score')
    plt.xticks(range(len(param_grid['max_depth'])), param_grid['max_depth'])
    plt.yticks(range(len(param_grid['n_estimators'])), param_grid['n_estimators'])
    plt.xlabel('Max Depth')
    plt.ylabel('Number of Estimators')
    plt.title('Grid Search Results')
    plt.savefig('assets/grid_search_results.png')
    plt.show()

plot_grid_search_results(grid_search)
```

### 2. Random Search

This is like trying random combinations of settings, which can be more efficient than trying every possible combination.

```python
from sklearn.model_selection import RandomizedSearchCV

# Define parameter distributions
param_dist = {
    'n_estimators': np.random.randint(50, 300, 10),
    'max_depth': [None] + list(np.random.randint(5, 40, 5)),
    'min_samples_split': np.random.randint(2, 20, 5)
}

# Perform random search
random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy'
)
random_search.fit(X, y)

# Visualize results
def plot_random_search_results(random_search):
    results = random_search.cv_results_
    plt.figure(figsize=(10, 6))
    plt.scatter(results['param_n_estimators'], results['mean_test_score'])
    plt.xlabel('Number of Estimators')
    plt.ylabel('Mean CV Score')
    plt.title('Random Search Results')
    plt.savefig('assets/random_search_results.png')
    plt.show()

plot_random_search_results(random_search)
```

### 3. Bayesian Optimization

This is like having a smart assistant that learns from previous attempts and suggests the most promising settings to try next.

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

def bayesian_optimization(X, y, n_iterations=20):
    # Define parameter space
    param_space = {
        'n_estimators': (50, 300),
        'max_depth': (5, 40),
        'min_samples_split': (2, 20)
    }
    
    # Initialize GP
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel)
    
    # Store results
    X_observed = []
    y_observed = []
    
    for i in range(n_iterations):
        # Sample parameters
        params = {
            'n_estimators': np.random.randint(param_space['n_estimators'][0], 
                                            param_space['n_estimators'][1]),
            'max_depth': np.random.randint(param_space['max_depth'][0], 
                                         param_space['max_depth'][1]),
            'min_samples_split': np.random.randint(param_space['min_samples_split'][0], 
                                                 param_space['min_samples_split'][1])
        }
        
        # Evaluate model
        model = RandomForestClassifier(**params)
        score = np.mean(cross_val_score(model, X, y, cv=5))
        
        X_observed.append(list(params.values()))
        y_observed.append(score)
        
        # Update GP
        gp.fit(np.array(X_observed), np.array(y_observed))
    
    return X_observed, y_observed

# Run Bayesian optimization
X_observed, y_observed = bayesian_optimization(X, y)

# Visualize results
def plot_bayesian_optimization(X_observed, y_observed):
    plt.figure(figsize=(10, 6))
    plt.plot(y_observed, 'bo-')
    plt.xlabel('Iteration')
    plt.ylabel('Mean CV Score')
    plt.title('Bayesian Optimization Progress')
    plt.savefig('assets/bayesian_optimization.png')
    plt.show()

plot_bayesian_optimization(X_observed, y_observed)
```

## Common Mistakes to Avoid ⚠️

1. **Overfitting to Validation Set**
   - Using too many hyperparameters
   - Not using cross-validation
   - Not having a separate test set

2. **Computational Resources**
   - Not considering time constraints
   - Not using parallel processing
   - Not using early stopping

3. **Parameter Space**
   - Using too narrow ranges
   - Using too wide ranges
   - Not considering parameter dependencies

## Practical Example: Credit Risk Prediction 💳

Let's see how hyperparameter tuning helps in a real-world scenario:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create credit risk dataset
np.random.seed(42)
n_samples = 1000

# Generate features
age = np.random.normal(35, 10, n_samples)
income = np.random.exponential(50000, n_samples)
credit_score = np.random.normal(700, 100, n_samples)

X = np.column_stack([age, income, credit_score])
y = (credit_score + income/1000 + age > 800).astype(int)  # Binary target

# Create pipeline with hyperparameter tuning
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Define parameter grid
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

# Perform grid search
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")
```

## Best Practices 🌟

### 1. Choosing the Right Tuning Method

```python
def compare_tuning_methods(X, y):
    # Grid Search
    grid_search = GridSearchCV(
        RandomForestClassifier(),
        param_grid,
        cv=5,
        scoring='accuracy'
    )
    grid_search.fit(X, y)
    
    # Random Search
    random_search = RandomizedSearchCV(
        RandomForestClassifier(),
        param_dist,
        n_iter=20,
        cv=5,
        scoring='accuracy'
    )
    random_search.fit(X, y)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(['Grid Search', 'Random Search'], 
            [grid_search.best_score_, random_search.best_score_])
    plt.ylabel('Best CV Score')
    plt.title('Comparison of Tuning Methods')
    plt.savefig('assets/tuning_methods_comparison.png')
    plt.show()

compare_tuning_methods(X, y)
```

## Additional Resources 📚

1. **Online Courses**
   - Coursera: Machine Learning by Andrew Ng
   - edX: Introduction to Machine Learning

2. **Books**
   - "Introduction to Machine Learning with Python" by Andreas Müller
   - "Hands-On Machine Learning with Scikit-Learn" by Aurélien Géron

3. **Documentation**
   - [Scikit-learn Grid Search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
   - [Scikit-learn Random Search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)

## Next Steps 🚀

Ready to learn more? Check out:

1. [Model Metrics](./metrics.md) to understand different ways to evaluate your model
2. [Model Selection](./model-selection.md) to choose the best model for your problem
3. [Scikit-learn Pipelines](./sklearn-pipelines.md) to streamline your machine learning workflow
