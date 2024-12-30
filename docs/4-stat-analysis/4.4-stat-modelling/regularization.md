# Regularization Techniques

## Introduction
Regularization helps prevent overfitting by adding penalty terms to the model's loss function. These techniques provide a systematic way to control model complexity and improve generalization performance.

### Mathematical Foundation
In regularized regression, we modify the standard loss function by adding penalty terms. The general form is:

$$ \min_{\beta} \left\{ \sum_{i=1}^n (y_i - \mathbf{x}_i^T\beta)^2 + \lambda \cdot \text{penalty}(\beta) \right\} $$

where:
- $\sum_{i=1}^n (y_i - \mathbf{x}_i^T\beta)^2$ is the standard least squares loss
- $\lambda$ is the regularization parameter
- $\text{penalty}(\beta)$ is the regularization term

Common regularization methods include:

1. Ridge Regression (L2):
$$ \min_{\beta} \left\{ \sum_{i=1}^n (y_i - \mathbf{x}_i^T\beta)^2 + \lambda \sum_{j=1}^p \beta_j^2 \right\} $$

2. LASSO (L1):
$$ \min_{\beta} \left\{ \sum_{i=1}^n (y_i - \mathbf{x}_i^T\beta)^2 + \lambda \sum_{j=1}^p |\beta_j| \right\} $$

3. Elastic Net (L1 + L2):
$$ \min_{\beta} \left\{ \sum_{i=1}^n (y_i - \mathbf{x}_i^T\beta)^2 + \lambda_1 \sum_{j=1}^p |\beta_j| + \lambda_2 \sum_{j=1}^p \beta_j^2 \right\} $$

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

def generate_sample_data(n=100, p=20, noise=0.1, seed=42):
    """Generate sample data with many features"""
    np.random.seed(seed)
    X = np.random.normal(0, 1, (n, p))
    # Only first 5 features are relevant
    true_beta = np.zeros(p)
    true_beta[:5] = [1, 0.8, 0.6, 0.4, 0.2]
    y = X @ true_beta + np.random.normal(0, noise, n)
    return X, y
```

## Ridge Regression (L2)

### 1. Understanding Ridge Regression
```python
def demonstrate_ridge_penalty():
    """Visualize the effect of Ridge penalty"""
    alphas = [0, 0.1, 1.0, 10.0]
    X, y = generate_sample_data()
    
    plt.figure(figsize=(12, 4))
    for i, alpha in enumerate(alphas, 1):
        model = Ridge(alpha=alpha)
        model.fit(X, y)
        
        plt.subplot(1, 4, i)
        plt.stem(model.coef_)
        plt.title(f'α = {alpha}')
        plt.xlabel('Feature')
        plt.ylabel('Coefficient')
    
    plt.tight_layout()
    plt.savefig('ridge_penalty.png')
    plt.close()
```

### 2. Implementing Ridge Regression
```python
def fit_ridge_regression(X, y, alphas=np.logspace(-3, 3, 100)):
    """Fit Ridge regression with different alpha values"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit models
    train_scores = []
    test_scores = []
    coef_paths = []
    
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)
        
        train_scores.append(model.score(X_train_scaled, y_train))
        test_scores.append(model.score(X_test_scaled, y_test))
        coef_paths.append(model.coef_)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Coefficient paths
    plt.subplot(121)
    coef_paths = np.array(coef_paths)
    for i in range(coef_paths.shape[1]):
        plt.plot(alphas, coef_paths[:, i])
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Coefficient Value')
    plt.title('Ridge Coefficient Paths')
    
    # R-squared scores
    plt.subplot(122)
    plt.plot(alphas, train_scores, label='Train')
    plt.plot(alphas, test_scores, label='Test')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('R² Score')
    plt.title('Ridge Model Performance')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ridge_analysis.png')
    plt.close()
    
    return {
        'alphas': alphas,
        'train_scores': train_scores,
        'test_scores': test_scores,
        'coef_paths': coef_paths
    }
```

## LASSO Regression (L1)

### 1. Understanding LASSO
```python
def demonstrate_lasso_penalty():
    """Visualize the effect of LASSO penalty"""
    alphas = [0, 0.01, 0.1, 1.0]
    X, y = generate_sample_data()
    
    plt.figure(figsize=(12, 4))
    for i, alpha in enumerate(alphas, 1):
        model = Lasso(alpha=alpha)
        model.fit(X, y)
        
        plt.subplot(1, 4, i)
        plt.stem(model.coef_)
        plt.title(f'α = {alpha}')
        plt.xlabel('Feature')
        plt.ylabel('Coefficient')
    
    plt.tight_layout()
    plt.savefig('lasso_penalty.png')
    plt.close()
```

### 2. Implementing LASSO
```python
def fit_lasso_regression(X, y, alphas=np.logspace(-3, 1, 100)):
    """Fit LASSO regression with different alpha values"""
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit models
    train_scores = []
    test_scores = []
    n_nonzero = []
    coef_paths = []
    
    for alpha in alphas:
        model = Lasso(alpha=alpha)
        model.fit(X_train_scaled, y_train)
        
        train_scores.append(model.score(X_train_scaled, y_train))
        test_scores.append(model.score(X_test_scaled, y_test))
        n_nonzero.append(np.sum(model.coef_ != 0))
        coef_paths.append(model.coef_)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Coefficient paths
    plt.subplot(131)
    coef_paths = np.array(coef_paths)
    for i in range(coef_paths.shape[1]):
        plt.plot(alphas, coef_paths[:, i])
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Coefficient Value')
    plt.title('LASSO Coefficient Paths')
    
    # R-squared scores
    plt.subplot(132)
    plt.plot(alphas, train_scores, label='Train')
    plt.plot(alphas, test_scores, label='Test')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('R² Score')
    plt.title('LASSO Model Performance')
    plt.legend()
    
    # Number of non-zero coefficients
    plt.subplot(133)
    plt.plot(alphas, n_nonzero)
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Number of Non-zero Coefficients')
    plt.title('Feature Selection')
    
    plt.tight_layout()
    plt.savefig('lasso_analysis.png')
    plt.close()
    
    return {
        'alphas': alphas,
        'train_scores': train_scores,
        'test_scores': test_scores,
        'n_nonzero': n_nonzero,
        'coef_paths': coef_paths
    }
```

## Elastic Net

### 1. Combining L1 and L2
```python
def fit_elastic_net(X, y, l1_ratios=[0.1, 0.5, 0.7, 0.9], 
                    alphas=np.logspace(-3, 1, 20)):
    """Fit Elastic Net with different parameters"""
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = []
    
    for l1_ratio in l1_ratios:
        scores = []
        for alpha in alphas:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
            model.fit(X_train_scaled, y_train)
            
            test_score = model.score(X_test_scaled, y_test)
            n_nonzero = np.sum(model.coef_ != 0)
            
            scores.append({
                'alpha': alpha,
                'test_score': test_score,
                'n_nonzero': n_nonzero
            })
        
        results.append({
            'l1_ratio': l1_ratio,
            'scores': scores
        })
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Test scores
    plt.subplot(121)
    for res in results:
        scores = pd.DataFrame(res['scores'])
        plt.plot(scores['alpha'], scores['test_score'], 
                label=f'L1 ratio = {res["l1_ratio"]}')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Test R² Score')
    plt.title('Elastic Net Performance')
    plt.legend()
    
    # Number of features
    plt.subplot(122)
    for res in results:
        scores = pd.DataFrame(res['scores'])
        plt.plot(scores['alpha'], scores['n_nonzero'], 
                label=f'L1 ratio = {res["l1_ratio"]}')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Number of Non-zero Coefficients')
    plt.title('Feature Selection')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('elastic_net_analysis.png')
    plt.close()
    
    return results
```

## Model Selection with Cross-Validation

### 1. Cross-Validation for Regularization
```python
from sklearn.model_selection import GridSearchCV

def select_regularization_params(X, y, model_type='ridge'):
    """Select best regularization parameters using cross-validation"""
    # Prepare parameter grid
    if model_type.lower() == 'ridge':
        model = Ridge()
        param_grid = {'alpha': np.logspace(-3, 3, 20)}
    elif model_type.lower() == 'lasso':
        model = Lasso()
        param_grid = {'alpha': np.logspace(-3, 1, 20)}
    else:  # elastic net
        model = ElasticNet()
        param_grid = {
            'alpha': np.logspace(-3, 1, 10),
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    
    # Perform grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='neg_mean_squared_error'
    )
    grid_search.fit(X, y)
    
    # Plot results
    results = pd.DataFrame(grid_search.cv_results_)
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        results['param_alpha'], 
        -results['mean_test_score'],
        yerr=results['std_test_score']
    )
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Mean Squared Error')
    plt.title(f'{model_type} Cross-validation Results')
    plt.savefig('cv_results.png')
    plt.close()
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': -grid_search.best_score_,
        'results': results
    }
```

## Practice Questions
1. When should you use Ridge vs LASSO regression?
2. How do you choose the regularization parameter?
3. What are the advantages of Elastic Net?
4. How does regularization help with multicollinearity?
5. What role does feature scaling play in regularization?

## Key Takeaways
1. Regularization prevents overfitting
2. Different penalties have different effects
3. Cross-validation helps select parameters
4. Feature scaling is important
5. Consider interpretability vs performance
