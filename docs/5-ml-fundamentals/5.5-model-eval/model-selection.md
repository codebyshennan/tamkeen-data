# Model Selection

## What is Model Selection?

Think of model selection like choosing the right tool for a job. Just as you wouldn't use a hammer to screw in a bolt, you need to choose the right machine learning model for your specific problem. Model selection helps us find the best model that balances performance, complexity, and practical considerations.

### Why Model Selection Matters

Imagine you're planning a road trip. You wouldn't just pick any vehicle - you'd consider factors like:

- How many people are traveling?
- What's the terrain like?
- What's your budget?
- How much luggage do you have?

Similarly, in machine learning, we need to consider:

- The type of problem (classification, regression, etc.)
- The size and nature of the data
- Computational resources
- Business requirements

## Real-World Analogies

### The Restaurant Menu Analogy

Think of model selection like choosing from a restaurant menu:

- Each dish (model) has different ingredients (features)
- Some dishes are quick to prepare (simple models)
- Others take more time but are more complex (complex models)
- You need to consider dietary restrictions (constraints)
- You want the best value for money (performance vs. cost)

### The Sports Team Analogy

Model selection is like building a sports team:

- Each player (model) has different strengths
- Some players are versatile (general-purpose models)
- Others are specialists (domain-specific models)
- You need to consider team chemistry (model ensemble)
- You want the best performance within your budget

## Types of Models

### 1. Linear Models

These are like following a straight path - simple and interpretable.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create sample dataset
X, y = make_classification(n_samples=1000, n_features=20, 
                         n_informative=15, n_redundant=5,
                         random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train linear model
linear_model = LogisticRegression()
linear_model.fit(X_train, y_train)

# Make predictions
y_pred_linear = linear_model.predict(X_test)
print(f"Linear Model Accuracy: {accuracy_score(y_test, y_pred_linear):.3f}")

# Visualize decision boundary
def plot_decision_boundary(model, X, y):
    # Reduce to 2D for visualization
    X_2d = X[:, :2]
    model.fit(X_2d, y)
    
    # Create mesh grid
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    # Predict on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, alpha=0.8)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.savefig('assets/linear_decision_boundary.png')
    plt.show()

plot_decision_boundary(linear_model, X, y)
```

### 2. Tree-Based Models

These are like following a decision tree - more complex but often more powerful.

```python
from sklearn.ensemble import RandomForestClassifier

# Train tree-based model
tree_model = RandomForestClassifier()
tree_model.fit(X_train, y_train)

# Make predictions
y_pred_tree = tree_model.predict(X_test)
print(f"Tree Model Accuracy: {accuracy_score(y_test, y_pred_tree):.3f}")

# Visualize feature importance
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), 
               [f'Feature {i+1}' for i in indices], 
               rotation=45)
    plt.tight_layout()
    plt.savefig('assets/feature_importance.png')
    plt.show()

plot_feature_importance(tree_model, [f'Feature {i+1}' for i in range(X.shape[1])])
```

### 3. Neural Networks

These are like having multiple layers of decision-making - very powerful but more complex.

```python
from sklearn.neural_network import MLPClassifier

# Train neural network
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50))
nn_model.fit(X_train, y_train)

# Make predictions
y_pred_nn = nn_model.predict(X_test)
print(f"Neural Network Accuracy: {accuracy_score(y_test, y_pred_nn):.3f}")

# Visualize learning curve
def plot_learning_curve(model, X, y):
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, val_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('assets/learning_curve.png')
    plt.show()

plot_learning_curve(nn_model, X, y)
```

## Model Comparison

Let's compare different models:

```python
def compare_models(models, X_train, X_test, y_train, y_test):
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = accuracy_score(y_test, y_pred)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('assets/model_comparison.png')
    plt.show()
    
    return results

# Compare models
models = {
    'Linear': LogisticRegression(),
    'Tree': RandomForestClassifier(),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50))
}

results = compare_models(models, X_train, X_test, y_train, y_test)
```

## Common Mistakes to Avoid

1. **Overfitting**
   - Using too complex models
   - Not using cross-validation
   - Not having enough data

2. **Underfitting**
   - Using too simple models
   - Not considering feature engineering
   - Not tuning hyperparameters

3. **Model Selection Bias**
   - Not considering business context
   - Not evaluating on new data
   - Not considering model interpretability

## Practical Example: Credit Risk Prediction

Let's see how different models perform on a credit risk prediction task:

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

# Create pipelines
pipelines = {
    'Linear': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ]),
    'Tree': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ]),
    'Neural Network': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', MLPClassifier(hidden_layer_sizes=(100, 50)))
    ])
}

# Compare pipelines
results = compare_models(pipelines, X_train, X_test, y_train, y_test)
```

## Best Practices

### 1. Model Selection Process

```python
def model_selection_process(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define models
    models = {
        'Linear': LogisticRegression(),
        'Tree': RandomForestClassifier(),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50))
    }
    
    # Compare models
    results = compare_models(models, X_train, X_test, y_train, y_test)
    
    # Plot learning curves for best model
    best_model_name = max(results, key=results.get)
    plot_learning_curve(models[best_model_name], X, y)
    
    return results

model_selection_process(X, y)
```

## Additional Resources

1. **Online Courses**
   - Coursera: Machine Learning by Andrew Ng
   - edX: Introduction to Machine Learning

2. **Books**
   - "Introduction to Machine Learning with Python" by Andreas Müller
   - "Hands-On Machine Learning with Scikit-Learn" by Aurélien Géron

3. **Documentation**
   - [Scikit-learn Model Selection](https://scikit-learn.org/stable/model_selection.html)
   - [Model Comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)

## Next Steps

Ready to learn more? Check out:

1. [Cross Validation](./cross-validation.md) to properly evaluate your model
2. [Hyperparameter Tuning](./hyperparameter-tuning.md) to optimize your model's performance
3. [Model Metrics](./metrics.md) to understand different ways to evaluate your model
