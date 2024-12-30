# Correlation Analysis

## Understanding Correlation

### What is Correlation?
Correlation measures the strength and direction of the relationship between two variables. It ranges from -1 (perfect negative correlation) to +1 (perfect positive correlation).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def demonstrate_correlations():
    """Show different correlation levels"""
    np.random.seed(42)
    n = 100
    
    # Generate correlated data
    x = np.random.normal(0, 1, n)
    
    # Different correlation levels
    y_strong_pos = x * 0.9 + np.random.normal(0, 0.3, n)  # Strong positive
    y_moderate = x * 0.5 + np.random.normal(0, 0.7, n)    # Moderate
    y_strong_neg = -x * 0.9 + np.random.normal(0, 0.3, n) # Strong negative
    
    plt.figure(figsize=(15, 4))
    
    # Strong positive correlation
    plt.subplot(131)
    plt.scatter(x, y_strong_pos)
    plt.title(f'Strong Positive\nr = {np.corrcoef(x, y_strong_pos)[0,1]:.2f}')
    
    # Moderate correlation
    plt.subplot(132)
    plt.scatter(x, y_moderate)
    plt.title(f'Moderate\nr = {np.corrcoef(x, y_moderate)[0,1]:.2f}')
    
    # Strong negative correlation
    plt.subplot(133)
    plt.scatter(x, y_strong_neg)
    plt.title(f'Strong Negative\nr = {np.corrcoef(x, y_strong_neg)[0,1]:.2f}')
    
    plt.tight_layout()
    plt.savefig('correlation_types.png')
    plt.close()
```

## Types of Correlation

### 1. Pearson Correlation
Measures linear relationships between continuous variables:

```python
def pearson_correlation(x, y):
    """Calculate and interpret Pearson correlation"""
    r, p_value = stats.pearsonr(x, y)
    
    # Interpret strength
    if abs(r) < 0.3:
        strength = "weak"
    elif abs(r) < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
    
    # Interpret direction
    direction = "positive" if r > 0 else "negative"
    
    return {
        'coefficient': r,
        'p_value': p_value,
        'strength': strength,
        'direction': direction,
        'interpretation': f"A {strength} {direction} correlation (r={r:.2f}, p={p_value:.4f})"
    }
```

### 2. Spearman Correlation
For monotonic relationships and ordinal data:

```python
def spearman_correlation(x, y):
    """Calculate and interpret Spearman correlation"""
    rho, p_value = stats.spearmanr(x, y)
    
    # Interpret strength
    if abs(rho) < 0.3:
        strength = "weak"
    elif abs(rho) < 0.7:
        strength = "moderate"
    else:
        strength = "strong"
    
    # Interpret direction
    direction = "positive" if rho > 0 else "negative"
    
    return {
        'coefficient': rho,
        'p_value': p_value,
        'strength': strength,
        'direction': direction,
        'interpretation': f"A {strength} {direction} rank correlation (ρ={rho:.2f}, p={p_value:.4f})"
    }
```

### 3. Kendall's Tau
For ordinal data and small sample sizes:

```python
def kendall_correlation(x, y):
    """Calculate and interpret Kendall's Tau correlation"""
    tau, p_value = stats.kendalltau(x, y)
    
    # Interpret strength
    if abs(tau) < 0.2:
        strength = "weak"
    elif abs(tau) < 0.5:
        strength = "moderate"
    else:
        strength = "strong"
    
    # Interpret direction
    direction = "positive" if tau > 0 else "negative"
    
    return {
        'coefficient': tau,
        'p_value': p_value,
        'strength': strength,
        'direction': direction,
        'interpretation': f"A {strength} {direction} rank correlation (τ={tau:.2f}, p={p_value:.4f})"
    }
```

## Visualization Techniques

### 1. Scatter Plots
```python
def plot_correlation_scatter(x, y, title="Correlation Scatter Plot"):
    """Create scatter plot with correlation line"""
    plt.figure(figsize=(8, 6))
    
    # Plot points
    plt.scatter(x, y, alpha=0.5)
    
    # Add correlation line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", alpha=0.8)
    
    # Add correlation coefficient
    r = np.corrcoef(x, y)[0,1]
    plt.title(f"{title}\nr = {r:.2f}")
    
    plt.tight_layout()
    plt.savefig('correlation_scatter.png')
    plt.close()
```

### 2. Correlation Matrix
```python
def plot_correlation_matrix(data):
    """Create correlation matrix heatmap"""
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                vmin=-1, 
                vmax=1, 
                center=0)
    
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
```

## Statistical Significance

### Testing Correlation Significance
```python
def test_correlation_significance(x, y, alpha=0.05):
    """Test significance of correlation"""
    # Calculate all three correlation types
    pearson = pearson_correlation(x, y)
    spearman = spearman_correlation(x, y)
    kendall = kendall_correlation(x, y)
    
    results = pd.DataFrame({
        'Coefficient': [pearson['coefficient'], 
                       spearman['coefficient'], 
                       kendall['coefficient']],
        'P-value': [pearson['p_value'], 
                    spearman['p_value'], 
                    kendall['p_value']],
        'Significant': [p < alpha for p in [pearson['p_value'], 
                                          spearman['p_value'], 
                                          kendall['p_value']]]
    }, index=['Pearson', 'Spearman', 'Kendall'])
    
    return results
```

## Common Issues and Solutions

### 1. Outliers
```python
def handle_outliers(x, y, threshold=3):
    """Handle outliers in correlation analysis"""
    # Z-score method
    z_scores = np.abs(stats.zscore(np.vstack([x, y]).T))
    
    # Mask for non-outlier points
    mask = (z_scores < threshold).all(axis=1)
    
    # Calculate correlation with and without outliers
    full_corr = np.corrcoef(x, y)[0,1]
    clean_corr = np.corrcoef(x[mask], y[mask])[0,1]
    
    return {
        'full_correlation': full_corr,
        'clean_correlation': clean_corr,
        'n_outliers': len(x) - sum(mask),
        'mask': mask
    }
```

### 2. Non-linearity
```python
def check_nonlinearity(x, y):
    """Check for non-linear relationships"""
    # Linear correlation
    linear_r = stats.pearsonr(x, y)[0]
    
    # Fit polynomial
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    
    # R-squared for polynomial fit
    y_pred = p(x)
    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {
        'linear_r': linear_r,
        'polynomial_r_squared': r_squared,
        'potentially_nonlinear': r_squared > linear_r**2
    }
```

## Best Practices

1. **Always Visualize First**
```python
def correlation_analysis_workflow(x, y):
    """Complete correlation analysis workflow"""
    # 1. Visual inspection
    plot_correlation_scatter(x, y)
    
    # 2. Check for non-linearity
    nonlinearity = check_nonlinearity(x, y)
    
    # 3. Handle outliers
    outlier_results = handle_outliers(x, y)
    
    # 4. Calculate correlations
    correlations = test_correlation_significance(x, y)
    
    return {
        'nonlinearity_check': nonlinearity,
        'outlier_analysis': outlier_results,
        'correlations': correlations
    }
```

2. **Consider Multiple Correlation Types**
3. **Check Assumptions**
4. **Report Effect Sizes**
5. **Consider Context**

## Practice Questions
1. When should you use Spearman vs Pearson correlation?
2. How do outliers affect correlation coefficients?
3. What does statistical significance mean for correlation?
4. How can you identify non-linear relationships?
5. What are the limitations of correlation analysis?

## Key Takeaways
1. Correlation ≠ causation
2. Always visualize relationships
3. Consider multiple correlation types
4. Check for outliers and non-linearity
5. Report both coefficient and significance
