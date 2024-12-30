# Model Selection

## Introduction
Model selection is the process of choosing the best model from a set of candidate models. This involves balancing model complexity with predictive performance through rigorous statistical criteria and validation techniques.

### Theoretical Foundation
The fundamental challenge in model selection is the bias-variance tradeoff. Given a true function f(x) and a model f̂(x), the expected prediction error can be decomposed as:

$$E[(Y - f̂(X))^2] = Bias[f̂(X)]^2 + Var[f̂(X)] + σ^2$$

where:
- Bias[f̂(X)] = E[f̂(X)] - f(X) is the systematic error
- Var[f̂(X)] is the variance of the prediction
- σ² is the irreducible error

### Information Criteria
Information criteria provide a theoretical framework for model selection by penalizing model complexity:

1. Akaike Information Criterion (AIC):
$$AIC = -2ln(L) + 2k$$

2. Bayesian Information Criterion (BIC):
$$BIC = -2ln(L) + kln(n)$$

where:
- L is the maximum likelihood
- k is the number of parameters
- n is the sample size

3. Adjusted R-squared:
$$R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-k-1}$$

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.linear_model import LinearRegression, Ridge, Lasso

def generate_sample_data(n=100, p=5, seed=42):
    """Generate sample data with multiple features"""
    np.random.seed(seed)
    X = np.random.normal(0, 1, (n, p))
    true_beta = np.array([1, 0.5, 0.2, 0, 0])  # Some coefficients are 0
    y = X @ true_beta + np.random.normal(0, 0.1, n)
    return X, y
```

## Information Criteria

### 1. AIC and BIC
```python
def compare_information_criteria(models):
    """Compare models using AIC and BIC"""
    results = pd.DataFrame({
        'AIC': [model.aic for model in models],
        'BIC': [model.bic for model in models]
    })
    
    # Plot criteria
    plt.figure(figsize=(10, 5))
    
    plt.subplot(121)
    plt.plot(results.index, results['AIC'], 'o-')
    plt.title('AIC by Model')
    plt.xlabel('Model')
    plt.ylabel('AIC')
    
    plt.subplot(122)
    plt.plot(results.index, results['BIC'], 'o-')
    plt.title('BIC by Model')
    plt.xlabel('Model')
    plt.ylabel('BIC')
    
    plt.tight_layout()
    plt.savefig('information_criteria.png')
    plt.close()
    
    return results
```

### 2. Adjusted R-squared
```python
def compare_r_squared(models, X, y):
    """Compare models using R-squared and adjusted R-squared"""
    results = pd.DataFrame({
        'R2': [model.rsquared for model in models],
        'Adj_R2': [model.rsquared_adj for model in models]
    })
    
    # Plot R-squared measures
    plt.figure(figsize=(8, 6))
    results.plot(marker='o')
    plt.title('R-squared Measures by Model')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('r_squared_comparison.png')
    plt.close()
    
    return results
```

## Cross-Validation

### 1. K-Fold Cross-Validation
```python
def perform_cross_validation(models, X, y, cv=5):
    """Perform k-fold cross-validation for multiple models"""
    results = []
    
    for model in models:
        scores = cross_val_score(model, X, y, cv=cv, 
                               scoring='neg_mean_squared_error')
        mse_scores = -scores  # Convert back to MSE
        
        results.append({
            'mean_mse': np.mean(mse_scores),
            'std_mse': np.std(mse_scores),
            'scores': mse_scores
        })
    
    # Plot results
    plt.figure(figsize=(10, 6))
    means = [r['mean_mse'] for r in results]
    stds = [r['std_mse'] for r in results]
    plt.errorbar(range(len(models)), means, yerr=stds, fmt='o-')
    plt.title('Cross-validation Results')
    plt.xlabel('Model')
    plt.ylabel('Mean Squared Error')
    plt.savefig('cross_validation.png')
    plt.close()
    
    return results
```

### 2. Time Series Cross-Validation
```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_cv(model, X, y, n_splits=5):
    """Perform time series cross-validation"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    plt.figure(figsize=(15, 5))
    
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
        
        # Plot this fold
        plt.subplot(1, n_splits, i+1)
        plt.plot(y, 'b-', label='Data')
        plt.axvline(x=len(y_train), color='r', linestyle='--')
        plt.title(f'Fold {i+1}')
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('time_series_cv.png')
    plt.close()
    
    return scores
```

## Feature Selection

### 1. Forward Selection
```python
def forward_selection(X, y, threshold=0.05):
    """Perform forward stepwise selection"""
    features = list(range(X.shape[1]))
    selected = []
    current_score = float('-inf')
    
    while features:
        scores = []
        for feature in features:
            test_features = selected + [feature]
            X_subset = X[:, test_features]
            model = sm.OLS(y, sm.add_constant(X_subset)).fit()
            scores.append({
                'feature': feature,
                'score': model.rsquared_adj,
                'pvalue': model.pvalues[-1]
            })
        
        best = max(scores, key=lambda x: x['score'])
        if best['pvalue'] < threshold and best['score'] > current_score:
            selected.append(best['feature'])
            features.remove(best['feature'])
            current_score = best['score']
        else:
            break
    
    return selected
```

### 2. LASSO Selection
```python
def lasso_selection(X, y, alphas=np.logspace(-4, 1, 50)):
    """Perform feature selection using LASSO"""
    coef_paths = []
    scores = []
    
    for alpha in alphas:
        model = Lasso(alpha=alpha)
        model.fit(X, y)
        coef_paths.append(model.coef_)
        scores.append(model.score(X, y))
    
    # Plot coefficient paths
    plt.figure(figsize=(10, 6))
    coef_paths = np.array(coef_paths)
    for i in range(coef_paths.shape[1]):
        plt.plot(alphas, coef_paths[:, i], label=f'Feature {i+1}')
    plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Coefficient Value')
    plt.title('LASSO Coefficient Paths')
    plt.legend()
    plt.savefig('lasso_paths.png')
    plt.close()
    
    return {
        'alphas': alphas,
        'coef_paths': coef_paths,
        'scores': scores
    }
```

## Model Comparison

### 1. Statistical Tests
```python
def compare_models_statistically(model1, model2):
    """Compare nested models using likelihood ratio test"""
    llf1 = model1.llf  # Log-likelihood of model 1
    llf2 = model2.llf  # Log-likelihood of model 2
    df1 = model1.df_model
    df2 = model2.df_model
    
    # Likelihood ratio test
    lr_stat = 2 * (llf2 - llf1)
    p_value = stats.chi2.sf(lr_stat, df2 - df1)
    
    return {
        'lr_statistic': lr_stat,
        'p_value': p_value,
        'better_model': 'Model 2' if p_value < 0.05 else 'Model 1'
    }
```

### 2. Prediction Performance
```python
def compare_prediction_performance(models, X, y):
    """Compare models based on prediction performance"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    results = []
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results.append({
            'r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'residuals': y_test - y_pred
        })
    
    # Plot residuals
    plt.figure(figsize=(10, 4))
    for i, res in enumerate(results):
        plt.subplot(1, len(models), i+1)
        plt.hist(res['residuals'], bins=20)
        plt.title(f'Model {i+1} Residuals')
    
    plt.tight_layout()
    plt.savefig('residual_comparison.png')
    plt.close()
    
    return results
```

## Practice Questions
1. How do you choose between AIC and BIC?
2. When should you use cross-validation vs holdout?
3. What are the trade-offs in feature selection methods?
4. How do you handle model selection with small datasets?
5. What role does domain knowledge play in model selection?

## Key Takeaways
1. Use multiple criteria for model selection
2. Consider model complexity vs performance
3. Validate results using cross-validation
4. Feature selection is crucial
5. Domain knowledge matters
