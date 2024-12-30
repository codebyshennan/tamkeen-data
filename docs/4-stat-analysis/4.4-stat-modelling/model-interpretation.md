# Model Interpretation

## Introduction
Understanding and interpreting statistical models is crucial for making informed decisions. This guide covers techniques for interpreting different types of models and their parameters.

### Mathematical Foundation

#### 1. Linear Models
For linear models, the interpretation is based on the equation:

$$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_px_p + \epsilon$$

where:
- β₀ is the intercept
- βᵢ represents the change in y for a one-unit increase in xᵢ, holding other variables constant
- ε ~ N(0, σ²) is the error term

#### 2. Logistic Models
For logistic regression, we interpret odds ratios:

$$logit(p) = ln(\frac{p}{1-p}) = \beta_0 + \beta_1x_1 + ... + \beta_px_p$$

The odds ratio for feature xᵢ is:

$$OR_i = e^{\beta_i}$$

#### 3. Standardized Coefficients
For comparing feature importance:

$$\beta_i^* = \beta_i \cdot \frac{σ_{x_i}}{σ_y}$$

where:
- β*ᵢ is the standardized coefficient
- σ_xᵢ is the standard deviation of feature i
- σ_y is the standard deviation of the target

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import shap

def generate_sample_data(n=1000, seed=42):
    """Generate sample data for interpretation examples"""
    np.random.seed(seed)
    X = pd.DataFrame({
        'age': np.random.normal(40, 10, n),
        'income': np.random.normal(50000, 15000, n),
        'credit_score': np.random.normal(700, 50, n),
        'debt_ratio': np.random.uniform(0.1, 0.6, n)
    })
    
    # Generate target (loan approval)
    z = (0.03 * X['age'] + 
         0.4 * (X['income']/10000) + 
         0.02 * (X['credit_score']-600) - 
         2 * X['debt_ratio'])
    prob = 1 / (1 + np.exp(-z))
    y = (np.random.random(n) < prob).astype(int)
    
    return X, y
```

## Coefficient Interpretation

### 1. Linear Regression Coefficients
```python
def interpret_linear_coefficients(X, y):
    """Interpret coefficients of linear regression"""
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Fit model
    model = sm.OLS(y, sm.add_constant(X_scaled)).fit()
    
    # Create coefficient plot
    coef_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.params[1:],
        'std_err': model.bse[1:],
        'p_value': model.pvalues[1:]
    })
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(coef_df['coefficient'], 
                range(len(coef_df)),
                xerr=1.96*coef_df['std_err'],
                fmt='o')
    plt.yticks(range(len(coef_df)), coef_df['feature'])
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Standardized Coefficients with 95% CI')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.savefig('linear_coefficients.png')
    plt.close()
    
    return coef_df
```

### 2. Logistic Regression Odds Ratios
```python
def interpret_logistic_odds_ratios(X, y):
    """Interpret odds ratios from logistic regression"""
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Fit model
    model = sm.Logit(y, sm.add_constant(X_scaled)).fit()
    
    # Calculate odds ratios and confidence intervals
    odds_ratios = pd.DataFrame({
        'feature': X.columns,
        'odds_ratio': np.exp(model.params[1:]),
        'ci_lower': np.exp(model.conf_int()[1:, 0]),
        'ci_upper': np.exp(model.conf_int()[1:, 1])
    })
    
    # Plot odds ratios
    plt.figure(figsize=(10, 6))
    plt.errorbar(odds_ratios['odds_ratio'],
                range(len(odds_ratios)),
                xerr=[odds_ratios['odds_ratio'] - odds_ratios['ci_lower'],
                      odds_ratios['ci_upper'] - odds_ratios['odds_ratio']],
                fmt='o')
    plt.yticks(range(len(odds_ratios)), odds_ratios['feature'])
    plt.axvline(x=1, color='r', linestyle='--')
    plt.xscale('log')
    plt.title('Odds Ratios with 95% CI')
    plt.xlabel('Odds Ratio (log scale)')
    plt.tight_layout()
    plt.savefig('odds_ratios.png')
    plt.close()
    
    return odds_ratios
```

## Feature Importance

### 1. SHAP Values
```python
def calculate_shap_values(X, y, model_type='linear'):
    """Calculate and visualize SHAP values"""
    # Prepare model
    if model_type == 'linear':
        model = LinearRegression()
    else:
        model = LogisticRegression()
    
    model.fit(X, y)
    
    # Calculate SHAP values
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    
    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()
    
    # Feature importance based on SHAP
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(shap_values).mean(0)
    }).sort_values('importance', ascending=False)
    
    return {
        'shap_values': shap_values,
        'importance': importance_df
    }
```

### 2. Partial Dependence Plots
```python
def plot_partial_dependence(X, y, feature, model_type='linear'):
    """Create partial dependence plot for a feature"""
    # Prepare model
    if model_type == 'linear':
        model = LinearRegression()
    else:
        model = LogisticRegression()
    
    model.fit(X, y)
    
    # Generate feature values
    feature_values = np.linspace(X[feature].min(), X[feature].max(), 100)
    predictions = []
    
    # Calculate predictions
    for value in feature_values:
        X_copy = X.copy()
        X_copy[feature] = value
        pred = model.predict(X_copy)
        predictions.append(np.mean(pred))
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(feature_values, predictions)
    plt.xlabel(feature)
    plt.ylabel('Predicted Value')
    plt.title(f'Partial Dependence Plot for {feature}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'pdp_{feature}.png')
    plt.close()
    
    return pd.DataFrame({
        'value': feature_values,
        'effect': predictions
    })
```

## Model Performance Analysis

### 1. Prediction Intervals
```python
def calculate_prediction_intervals(model, X, y, alpha=0.05):
    """Calculate prediction intervals for regression"""
    # Get predictions
    y_pred = model.predict(X)
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Calculate prediction intervals
    mse = np.mean(residuals**2)
    std_pred = np.sqrt(mse)
    
    z_score = stats.norm.ppf(1 - alpha/2)
    margin = z_score * std_pred
    
    return pd.DataFrame({
        'prediction': y_pred,
        'lower_bound': y_pred - margin,
        'upper_bound': y_pred + margin
    })
```

### 2. Calibration Analysis
```python
def analyze_calibration(y_true, y_prob, n_bins=10):
    """Analyze calibration of probability predictions"""
    # Calculate calibration curve
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    
    bin_sums = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_means = bin_sums / bin_counts
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot calibration curve
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    plt.plot(bin_centers, bin_means, 'o-', label='Model calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Frequency')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('calibration_plot.png')
    plt.close()
    
    return pd.DataFrame({
        'bin_center': bin_centers,
        'observed_freq': bin_means,
        'count': bin_counts
    })
```

## Interaction Effects

### 1. Visualizing Interactions
```python
def plot_interaction_effects(X, y, feature1, feature2):
    """Visualize interaction between two features"""
    # Fit model with interaction term
    X_interact = X.copy()
    X_interact['interaction'] = X[feature1] * X[feature2]
    
    model = LinearRegression()
    model.fit(X_interact, y)
    
    # Create interaction plot
    plt.figure(figsize=(10, 6))
    
    feature1_low = X[feature1] < X[feature1].median()
    feature1_high = X[feature1] >= X[feature1].median()
    
    plt.scatter(X[feature2][feature1_low], y[feature1_low], 
                alpha=0.5, label=f'Low {feature1}')
    plt.scatter(X[feature2][feature1_high], y[feature1_high], 
                alpha=0.5, label=f'High {feature1}')
    
    plt.xlabel(feature2)
    plt.ylabel('Target')
    plt.title(f'Interaction between {feature1} and {feature2}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('interaction_plot.png')
    plt.close()
```

## Practice Questions
1. How do you interpret standardized coefficients?
2. What's the difference between coefficients and odds ratios?
3. When should you use SHAP values vs coefficients?
4. How do you identify important interaction effects?
5. What role do prediction intervals play in interpretation?

## Key Takeaways
1. Consider both statistical and practical significance
2. Use multiple interpretation methods
3. Visualize relationships and effects
4. Account for uncertainty in interpretations
