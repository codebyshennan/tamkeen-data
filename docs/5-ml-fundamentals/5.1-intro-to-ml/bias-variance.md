# Understanding Bias-Variance Tradeoff 🎯

The bias-variance tradeoff is a fundamental concept in machine learning that helps us understand prediction errors and model complexity. Imagine you're learning to play darts:
- High Bias: Consistently hitting the same spot but far from the bullseye
- High Variance: Hitting all over the board with no consistency
- Perfect Balance: Consistently hitting close to the bullseye

## Mathematical Foundation 📐

### Error Decomposition
The expected prediction error can be decomposed into three parts:

$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2$$

Where:
- $y$ is the true value
- $\hat{f}(x)$ is our model's prediction
- $\sigma^2$ is irreducible error

### Bias Term
$$\text{Bias}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - f(x)$$
- Measures how far the model's predictions are from the true function
- High bias = Underfitting (model is too simple)

### Variance Term
$$\text{Var}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)^2] - \mathbb{E}[\hat{f}(x)]^2$$
- Measures how much model predictions vary for different training sets
- High variance = Overfitting (model is too complex)

## Visual Understanding 📊

Let's visualize different model complexities:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Generate sample data with noise
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
true_function = 3 * np.sin(X.ravel())
y = true_function + np.random.normal(0, 0.5, 100)

# Create models with different complexities
degrees = [1, 3, 15]  # Linear, Cubic, and High-degree polynomial
plt.figure(figsize=(15, 5))

for i, degree in enumerate(degrees, 1):
    # Create polynomial regression model
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    # Fit model
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Plot results
    plt.subplot(1, 3, i)
    plt.scatter(X, y, color='blue', alpha=0.5, label='Data')
    plt.plot(X, true_function, 'g--', label='True Function')
    plt.plot(X, y_pred, 'r-', label=f'Degree {degree}')
    plt.title(f'Polynomial Degree {degree}')
    plt.legend()

plt.tight_layout()
plt.show()
```

## Real-World Example: House Price Prediction 🏠

### Setup and Data Generation
```python
from sklearn.model_selection import train_test_split
import pandas as pd

# Generate realistic house price data
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'size': np.random.normal(2000, 500, n_samples),  # Square footage
    'bedrooms': np.random.randint(1, 6, n_samples),
    'age': np.random.randint(0, 50, n_samples),
    'location_score': np.random.uniform(0, 1, n_samples)
})

# True price function with some non-linearity
df['price'] = (
    200 * df['size'] 
    + 50000 * df['bedrooms'] 
    - 1000 * df['age'] 
    + 100000 * np.sin(5 * df['location_score'])  # Non-linear location effect
    + np.random.normal(0, 50000, n_samples)  # Random noise
)

# Split data
X = df[['size', 'bedrooms', 'age', 'location_score']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### Model Comparison
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class ModelAnalyzer:
    def __init__(self, name, model, X_train, X_test, y_train, y_test):
        self.name = name
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def fit_and_evaluate(self):
        # Fit model
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        results = {
            'train_r2': r2_score(self.y_train, train_pred),
            'test_r2': r2_score(self.y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, test_pred))
        }
        
        self._print_results(results)
        return results
    
    def _print_results(self, results):
        print(f"\n{self.name} Results:")
        print(f"Training R²: {results['train_r2']:.3f}")
        print(f"Testing R²: {results['test_r2']:.3f}")
        print(f"Training RMSE: ${results['train_rmse']:,.2f}")
        print(f"Testing RMSE: ${results['test_rmse']:,.2f}")
        print(f"R² Difference: {results['train_r2'] - results['test_r2']:.3f}")

# 1. High Bias Model (Too Simple)
high_bias_model = ModelAnalyzer(
    "High Bias Model",
    LinearRegression(),
    X_train[['size']], X_test[['size']],  # Using only size feature
    y_train, y_test
)

# 2. High Variance Model (Too Complex)
high_variance_model = ModelAnalyzer(
    "High Variance Model",
    RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_leaf=1),
    X_train, X_test,
    y_train, y_test
)

# 3. Balanced Model
balanced_model = ModelAnalyzer(
    "Balanced Model",
    RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=5),
    X_train, X_test,
    y_train, y_test
)

# Evaluate all models
results = {
    'High Bias': high_bias_model.fit_and_evaluate(),
    'High Variance': high_variance_model.fit_and_evaluate(),
    'Balanced': balanced_model.fit_and_evaluate()
}
```

## Learning Curves Analysis 📈

Learning curves help diagnose bias and variance issues by showing how model performance changes with training data size.

```python
from sklearn.model_selection import learning_curve

def plot_learning_curves(model, X, y, title):
    """Plot learning curves with confidence intervals"""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='r2',
        n_jobs=-1
    )
    
    # Calculate means and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    
    # Plot means
    plt.plot(train_sizes, train_mean, label='Training score', color='blue')
    plt.plot(train_sizes, val_mean, label='Cross-validation score', color='red')
    
    # Plot confidence intervals
    plt.fill_between(train_sizes, 
                    train_mean - train_std,
                    train_mean + train_std, 
                    alpha=0.1, color='blue')
    plt.fill_between(train_sizes, 
                    val_mean - val_std,
                    val_mean + val_std, 
                    alpha=0.1, color='red')
    
    plt.title(title)
    plt.xlabel('Training Examples')
    plt.ylabel('R² Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Plot learning curves for each model type
models = {
    'High Bias': LinearRegression(),
    'High Variance': RandomForestRegressor(n_estimators=100, max_depth=None),
    'Balanced': RandomForestRegressor(n_estimators=100, max_depth=5)
}

for name, model in models.items():
    plot_learning_curves(model, 
                        X_train if name != 'High Bias' else X_train[['size']], 
                        y_train,
                        f'Learning Curves - {name} Model')
```

## Model Complexity Analysis 🔍

### Cross-Validation Across Complexities
```python
from sklearn.model_selection import cross_val_score

def analyze_model_complexity(X, y):
    """Analyze how model complexity affects performance"""
    max_depths = [2, 3, 4, 5, 6, 7, 8, None]
    cv_scores = []
    train_scores = []
    
    for depth in max_depths:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=depth,
            random_state=42
        )
        
        # Cross-validation score
        cv_score = cross_val_score(
            model, X, y,
            cv=5, scoring='r2'
        ).mean()
        
        # Training score
        model.fit(X, y)
        train_score = model.score(X, y)
        
        cv_scores.append(cv_score)
        train_scores.append(train_score)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(max_depths)), train_scores, 
             'o-', label='Training Score')
    plt.plot(range(len(max_depths)), cv_scores, 
             'o-', label='CV Score')
    plt.xticks(range(len(max_depths)), 
               [str(depth) if depth else 'None' for depth in max_depths])
    plt.xlabel('Max Depth')
    plt.ylabel('R² Score')
    plt.title('Model Complexity vs Performance')
    plt.legend()
    plt.grid(True)
    plt.show()

analyze_model_complexity(X, y)
```

## Diagnostic Tools 🔧

### 1. Bias Detection
```python
class BiasAnalyzer:
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def analyze(self):
        """Comprehensive bias analysis"""
        # Fit model
        self.model.fit(self.X_train, self.y_train)
        
        # Get predictions
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        train_score = r2_score(self.y_train, train_pred)
        test_score = r2_score(self.y_test, test_pred)
        
        print("Bias Analysis Results:")
        print(f"Training R²: {train_score:.3f}")
        print(f"Testing R²: {test_score:.3f}")
        print(f"Score Difference: {abs(train_score - test_score):.3f}")
        
        # Bias indicators
        if train_score < 0.5:
            print("⚠️ High bias suspected (low training score)")
        if abs(train_score - test_score) < 0.1:
            print("ℹ️ Similar train/test scores suggest underfitting")
```

### 2. Variance Detection
```python
class VarianceAnalyzer:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        
    def analyze(self):
        """Analyze prediction variance across different samples"""
        n_iterations = 5
        predictions = []
        
        for i in range(n_iterations):
            # Create bootstrap sample
            indices = np.random.choice(
                len(self.X), 
                size=len(self.X), 
                replace=True
            )
            X_sample = self.X.iloc[indices]
            y_sample = self.y.iloc[indices]
            
            # Fit model and predict
            self.model.fit(X_sample, y_sample)
            pred = self.model.predict(self.X)
            predictions.append(pred)
            
        # Calculate prediction variance
        predictions = np.array(predictions)
        variance = np.var(predictions, axis=0)
        
        print("Variance Analysis Results:")
        print(f"Mean prediction variance: {np.mean(variance):.2f}")
        print(f"Max prediction variance: {np.max(variance):.2f}")
        
        if np.mean(variance) > 1e6:
            print("⚠️ High variance suspected (large prediction variations)")
```

## Solutions and Best Practices 💡

### Fixing High Bias
1. Increase Model Complexity
   - Add more features: $$f(x_1, ..., x_n) \rightarrow f(x_1, ..., x_n, x_{n+1})$$
   - Use more complex algorithms
   - Reduce regularization: $$\min_{\theta} \left(\text{Loss}(\theta) + \lambda ||\theta||^2\right), \lambda \downarrow$$

2. Feature Engineering
   - Create interaction terms: $$x_{new} = x_1 \times x_2$$
   - Add polynomial features: $$x \rightarrow [x, x^2, x^3, ...]$$

### Fixing High Variance
1. Regularization
   - L1 (Lasso): $$\min_{\theta} \left(\text{Loss}(\theta) + \lambda \sum_{i=1}^n |\theta_i|\right)$$
   - L2 (Ridge): $$\min_{\theta} \left(\text{Loss}(\theta) + \lambda \sum_{i=1}^n \theta_i^2\right)$$

2. Data Augmentation
   - Increase training data
   - Add noise: $$x_{new} = x + \epsilon, \epsilon \sim \mathcal{N}(0, \sigma^2)$$

3. Ensemble Methods
   - Bagging: $$f_{bag}(x) = \frac{1}{M}\sum_{m=1}^M f_m(x)$$
   - Random Forest
   - Gradient Boosting

## Model Selection Checklist ✅

1. **Data Assessment**
   - [ ] Sufficient data quantity
   - [ ] Data quality and cleanliness
   - [ ] Feature relevance

2. **Model Complexity**
   - [ ] Start simple, increase complexity gradually
   - [ ] Monitor training and validation metrics
   - [ ] Use cross-validation

3. **Regularization**
   - [ ] Try different regularization methods
   - [ ] Tune regularization strength
   - [ ] Validate impact

4. **Validation Strategy**
   - [ ] Use proper train/test split
   - [ ] Implement k-fold cross-validation
   - [ ] Consider time-based splitting for temporal data

5. **Performance Monitoring**
   - [ ] Track both training and validation metrics
   - [ ] Plot learning curves
   - [ ] Analyze error distributions

## Next Steps 📚

Now that you understand the bias-variance tradeoff, practice these concepts with the [assignment](./assignment.md) and apply them to real-world problems!
