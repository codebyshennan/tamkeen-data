# Polynomial Regression

## Introduction

Polynomial regression is a powerful extension of linear regression that allows us to model non-linear relationships between variables. While linear regression assumes a straight-line relationship, polynomial regression can capture more complex patterns in the data.

### Real-world Examples

Let's look at some scenarios where polynomial regression is useful:

1. **Growth Patterns**
   - Plant growth over time
   - Population growth
   - Economic trends

2. **Physical Phenomena**
   - Projectile motion
   - Temperature changes
   - Chemical reactions

3. **Business Applications**
   - Sales trends
   - Customer behavior
   - Market saturation

### Visualizing Non-linear Relationships

Imagine you're studying how study time affects exam scores. The relationship might not be linear - there could be diminishing returns after a certain point:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
study_hours = np.linspace(0, 10, 100)
# Create a non-linear relationship
scores = 50 + 10*study_hours - 0.5*study_hours**2 + np.random.normal(0, 5, 100)

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(study_hours, scores, alpha=0.5)
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.title('Study Time vs Exam Score')
plt.grid(True)
plt.savefig('nonlinear_relationship.png')
plt.close()
```

## Understanding Polynomial Regression

### What Makes It Different from Linear Regression?

While linear regression uses a straight line (degree 1 polynomial), polynomial regression can use curves of higher degrees. Here's how they compare:

```python
def compare_linear_polynomial():
    """Compare linear and polynomial fits"""
    x = np.linspace(-3, 3, 100)
    y = x**3 - 2*x**2 + x + np.random.normal(0, 0.5, 100)
    
    # Fit linear regression
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(x.reshape(-1, 1), y)
    y_lin = lin_reg.predict(x.reshape(-1, 1))
    
    # Fit polynomial regression
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(x.reshape(-1, 1))
    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y)
    y_poly = poly_reg.predict(X_poly)
    
    # Plot both fits
    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, alpha=0.5, label='Data')
    plt.plot(x, y_lin, 'r-', label='Linear Fit')
    plt.plot(x, y_poly, 'g-', label='Polynomial Fit (degree=3)')
    plt.legend()
    plt.title('Linear vs Polynomial Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.savefig('linear_vs_polynomial.png')
    plt.close()
```

### The Polynomial Equation

A polynomial regression model of degree n can be written as:

$$y = \beta_0 + \beta_1x + \beta_2x^2 + ... + \beta_nx^n + \epsilon$$

Where:

- $y$ is the dependent variable
- $x$ is the independent variable
- $\beta_0, \beta_1, ..., \beta_n$ are the coefficients
- $\epsilon$ is the error term

### Choosing the Right Degree

The degree of the polynomial is crucial. Too low, and you underfit the data. Too high, and you overfit. Let's visualize this:

```python
def plot_different_degrees():
    """Show effect of different polynomial degrees"""
    x = np.linspace(-3, 3, 100)
    y = x**3 - 2*x**2 + x + np.random.normal(0, 0.5, 100)
    
    degrees = [1, 2, 3, 10]
    plt.figure(figsize=(15, 10))
    
    for i, degree in enumerate(degrees, 1):
        plt.subplot(2, 2, i)
        
        # Fit polynomial
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(x.reshape(-1, 1))
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        # Plot
        plt.scatter(x, y, alpha=0.5)
        plt.plot(x, y_pred, 'r-')
        plt.title(f'Degree {degree} Polynomial')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('polynomial_degrees.png')
    plt.close()
```

## Building a Polynomial Regression Model

### Step 1: Prepare the Data

```python
def prepare_polynomial_data(X, y, degree=2):
    """Transform data for polynomial regression"""
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    
    return X_scaled, poly, scaler
```

### Step 2: Train the Model

```python
def train_polynomial_model(X, y):
    """Train and return a polynomial regression model"""
    from sklearn.linear_model import LinearRegression
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model
```

### Step 3: Evaluate the Model

```python
def evaluate_polynomial_model(model, X, y):
    """Evaluate model performance"""
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.grid(True)
    plt.savefig('polynomial_predictions.png')
    plt.close()
    
    return {
        'mse': mse,
        'r2': r2,
        'predictions': y_pred
    }
```

## Common Challenges and Solutions

1. **Overfitting**
   - Problem: Model fits noise in the training data
   - Solution: Use cross-validation to select optimal degree
   - Example: Try different degrees and compare validation scores

2. **Multicollinearity**
   - Problem: High correlation between polynomial terms
   - Solution: Use regularization or orthogonal polynomials
   - Example: Ridge or Lasso regression

3. **Extrapolation**
   - Problem: Poor predictions outside training range
   - Solution: Be cautious with predictions beyond data range
   - Example: Add confidence intervals to predictions

## Practice Exercise

Try building a polynomial regression model to predict house prices based on square footage. Consider:

1. What degree polynomial might be appropriate?
2. How can you prevent overfitting?
3. How do you interpret the coefficients?
4. What metrics should you use to evaluate the model?

## Additional Resources

- [Scikit-learn Polynomial Features](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- [Introduction to Statistical Learning](https://www.statlearning.com/) (Chapter 7)
- [Polynomial Regression in Python](https://www.statsmodels.org/stable/examples/notebooks/generated/polynomial-regression.html)

Remember: The key to successful polynomial regression is finding the right balance between model complexity and generalization. Always validate your model on unseen data!
