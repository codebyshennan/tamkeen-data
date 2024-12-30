# Statistical Tests: Your Data Analysis Toolkit 🧰

## Introduction: Why Statistical Tests Matter 🎯
Think of statistical tests as your data detective tools - they help you uncover patterns, relationships, and differences that might not be obvious at first glance. Whether you're comparing customer groups, analyzing experimental results, or exploring relationships between variables, statistical tests help you make informed decisions based on evidence rather than intuition!

## The Statistical Tests Family Tree 🌳

### 1. T-Tests: Comparing Means 📊
Like comparing recipes by taste-testing - are they really different or just slightly varied?

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class TTestAnalyzer:
    """A comprehensive t-test analysis toolkit"""
    
    def __init__(self, data1, data2=None, population_mean=None):
        self.data1 = data1
        self.data2 = data2
        self.pop_mean = population_mean
        
    def run_all_tests(self):
        """Run all applicable t-tests"""
        results = {}
        
        # One-sample t-test
        if self.pop_mean is not None:
            results['one_sample'] = self._one_sample_test()
        
        # Two-sample tests
        if self.data2 is not None:
            results['independent'] = self._independent_test()
            if len(self.data1) == len(self.data2):
                results['paired'] = self._paired_test()
        
        return pd.DataFrame(results).T
    
    def _one_sample_test(self):
        """Perform one-sample t-test"""
        t_stat, p_val = stats.ttest_1samp(self.data1, self.pop_mean)
        effect_size = (np.mean(self.data1) - self.pop_mean) / np.std(self.data1)
        return {
            'test_type': 'One-sample t-test',
            't_statistic': t_stat,
            'p_value': p_val,
            'effect_size': effect_size
        }
    
    def _independent_test(self):
        """Perform independent t-test"""
        t_stat, p_val = stats.ttest_ind(self.data1, self.data2)
        effect_size = (np.mean(self.data1) - np.mean(self.data2)) / \
                     np.sqrt((np.var(self.data1) + np.var(self.data2)) / 2)
        return {
            'test_type': 'Independent t-test',
            't_statistic': t_stat,
            'p_value': p_val,
            'effect_size': effect_size
        }
    
    def _paired_test(self):
        """Perform paired t-test"""
        t_stat, p_val = stats.ttest_rel(self.data1, self.data2)
        effect_size = np.mean(self.data1 - self.data2) / np.std(self.data1 - self.data2)
        return {
            'test_type': 'Paired t-test',
            't_statistic': t_stat,
            'p_value': p_val,
            'effect_size': effect_size
        }
    
    def visualize(self):
        """Create comprehensive visualization"""
        plt.figure(figsize=(15, 5))
        
        # Distribution plot
        plt.subplot(131)
        if self.data2 is not None:
            sns.kdeplot(self.data1, label='Group 1')
            sns.kdeplot(self.data2, label='Group 2')
        else:
            sns.kdeplot(self.data1)
            plt.axvline(self.pop_mean, color='r', linestyle='--', 
                       label=f'Population Mean ({self.pop_mean})')
        plt.title('Distribution Comparison')
        plt.legend()
        
        # Box plot
        plt.subplot(132)
        if self.data2 is not None:
            sns.boxplot(data=[self.data1, self.data2])
            plt.xticks([0, 1], ['Group 1', 'Group 2'])
        else:
            sns.boxplot(data=self.data1)
        plt.title('Box Plot')
        
        # Q-Q plot
        plt.subplot(133)
        stats.probplot(self.data1, dist="norm", plot=plt)
        plt.title('Q-Q Plot (Group 1)')
        
        plt.tight_layout()
        plt.savefig('docs/4-stat-analysis/4.2-hypotheses-testing/assets/t_test_analysis.png')
        plt.close()
```

### 2. ANOVA: Comparing Multiple Groups 🎭
Like being a judge in a cooking competition - are any of these dishes significantly different?

```python
class ANOVAAnalyzer:
    """ANOVA analysis toolkit"""
    
    def __init__(self, *groups, group_names=None):
        self.groups = groups
        self.group_names = group_names or [f'Group {i+1}' for i in range(len(groups))]
        
    def analyze(self):
        """Perform comprehensive ANOVA analysis"""
        # One-way ANOVA
        f_stat, p_val = stats.f_oneway(*self.groups)
        
        # Effect size (eta-squared)
        df_between = len(self.groups) - 1
        df_total = sum(len(group) for group in self.groups) - 1
        grand_mean = np.mean([np.mean(group) for group in self.groups])
        
        ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 
                        for group in self.groups)
        ss_total = sum(sum((x - grand_mean)**2) for group in self.groups)
        eta_squared = ss_between / ss_total
        
        # Post-hoc tests
        from itertools import combinations
        posthoc = []
        for i, j in combinations(range(len(self.groups)), 2):
            t_stat, p_val = stats.ttest_ind(self.groups[i], self.groups[j])
            posthoc.append({
                'comparison': f'{self.group_names[i]} vs {self.group_names[j]}',
                't_statistic': t_stat,
                'p_value': p_val
            })
        
        return {
            'anova_results': {
                'f_statistic': f_stat,
                'p_value': p_val,
                'eta_squared': eta_squared
            },
            'posthoc_tests': pd.DataFrame(posthoc)
        }
    
    def visualize(self):
        """Create comprehensive visualization"""
        plt.figure(figsize=(15, 5))
        
        # Box plot
        plt.subplot(131)
        sns.boxplot(data=self.groups)
        plt.xticks(range(len(self.groups)), self.group_names)
        plt.title('Group Comparisons')
        
        # Violin plot
        plt.subplot(132)
        sns.violinplot(data=self.groups)
        plt.xticks(range(len(self.groups)), self.group_names)
        plt.title('Distribution Shapes')
        
        # Mean plot with error bars
        plt.subplot(133)
        means = [np.mean(group) for group in self.groups]
        sems = [stats.sem(group) for group in self.groups]
        plt.errorbar(range(len(self.groups)), means, yerr=sems, fmt='o')
        plt.xticks(range(len(self.groups)), self.group_names)
        plt.title('Means with Standard Errors')
        
        plt.tight_layout()
        plt.savefig('docs/4-stat-analysis/4.2-hypotheses-testing/assets/anova_analysis.png')
        plt.close()
```

### 3. Chi-Square Tests: Analyzing Categories 📊
Like checking if dice are fair - are the outcomes distributed as expected?

```python
class ChiSquareAnalyzer:
    """Chi-square analysis toolkit"""
    
    def __init__(self, observed, expected=None):
        self.observed = np.array(observed)
        self.expected = np.array(expected) if expected is not None else None
        
    def analyze(self):
        """Perform chi-square analysis"""
        if self.expected is None:
            # Goodness of fit test
            chi2, p_val = stats.chisquare(self.observed)
            test_type = 'Goodness of fit'
        else:
            # Test of independence
            chi2, p_val = stats.chisquare(self.observed, self.expected)
            test_type = 'Test of independence'
        
        # Effect size (Cramer's V)
        n = np.sum(self.observed)
        min_dim = min(self.observed.shape) - 1
        cramer_v = np.sqrt(chi2 / (n * min_dim))
        
        return {
            'test_type': test_type,
            'chi_square': chi2,
            'p_value': p_val,
            'cramer_v': cramer_v
        }
    
    def visualize(self):
        """Create visualization"""
        plt.figure(figsize=(12, 5))
        
        # Observed vs Expected
        plt.subplot(121)
        x = np.arange(len(self.observed))
        width = 0.35
        
        plt.bar(x - width/2, self.observed, width, label='Observed')
        if self.expected is not None:
            plt.bar(x + width/2, self.expected, width, label='Expected')
        
        plt.title('Observed vs Expected Frequencies')
        plt.legend()
        
        # Residuals
        if self.expected is not None:
            plt.subplot(122)
            residuals = self.observed - self.expected
            plt.bar(x, residuals)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('Residuals')
        
        plt.tight_layout()
        plt.savefig('docs/4-stat-analysis/4.2-hypotheses-testing/assets/chi_square_analysis.png')
        plt.close()
```

### 4. Correlation Tests: Measuring Relationships 🔄
Like checking if ice cream sales and temperature are related!

```python
class CorrelationAnalyzer:
    """Correlation analysis toolkit"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def analyze(self):
        """Perform multiple correlation tests"""
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(self.x, self.y)
        
        # Spearman correlation
        spearman_r, spearman_p = stats.spearmanr(self.x, self.y)
        
        # Kendall's Tau
        kendall_tau, kendall_p = stats.kendalltau(self.x, self.y)
        
        return pd.DataFrame({
            'correlation': [pearson_r, spearman_r, kendall_tau],
            'p_value': [pearson_p, spearman_p, kendall_p]
        }, index=['Pearson', 'Spearman', 'Kendall'])
    
    def visualize(self):
        """Create correlation visualizations"""
        plt.figure(figsize=(15, 5))
        
        # Scatter plot
        plt.subplot(131)
        sns.scatterplot(x=self.x, y=self.y)
        plt.title('Scatter Plot')
        
        # Regression plot
        plt.subplot(132)
        sns.regplot(x=self.x, y=self.y)
        plt.title('Regression Plot')
        
        # Hexbin plot for large datasets
        plt.subplot(133)
        plt.hexbin(self.x, self.y, gridsize=20)
        plt.colorbar(label='Count')
        plt.title('Hexbin Plot')
        
        plt.tight_layout()
        plt.savefig('docs/4-stat-analysis/4.2-hypotheses-testing/assets/correlation_analysis.png')
        plt.close()
```

## Choosing the Right Test 🎯

### The Test Selection Flowchart 📊

```python
def select_statistical_test(
    data_type='continuous',
    n_groups=2,
    paired=False,
    normal=True,
    sample_size=100
):
    """
    Guide for selecting appropriate statistical test
    
    Parameters:
    -----------
    data_type : str
        'continuous' or 'categorical'
    n_groups : int
        Number of groups to compare
    paired : bool
        Whether the data is paired
    normal : bool
        Whether the data is normally distributed
    sample_size : int
        Sample size
        
    Returns:
    --------
    dict with test recommendation and assumptions
    """
    if data_type == 'continuous':
        if n_groups == 1:
            test = "One-sample t-test" if normal else "Wilcoxon signed-rank test"
        elif n_groups == 2:
            if paired:
                test = "Paired t-test" if normal else "Wilcoxon signed-rank test"
            else:
                test = "Independent t-test" if normal else "Mann-Whitney U test"
        else:
            test = "ANOVA" if normal else "Kruskal-Wallis H-test"
    else:  # categorical
        if n_groups == 1:
            test = "Chi-square goodness of fit"
        elif n_groups == 2:
            test = "Fisher's exact test" if sample_size < 30 else "Chi-square test"
        else:
            test = "Chi-square test of independence"
    
    return {
        'recommended_test': test,
        'assumptions': {
            'normality_required': normal and data_type == 'continuous',
            'minimum_sample_size': sample_size,
            'paired_data': paired if n_groups == 2 else 'N/A'
        }
    }
```

## Checking Assumptions ✅

### 1. Normality Tests 📈

```python
class NormalityChecker:
    """Toolkit for checking normality assumptions"""
    
    def __init__(self, data):
        self.data = data
    
    def check_all(self):
        """Run comprehensive normality checks"""
        # Statistical tests
        shapiro_stat, shapiro_p = stats.shapiro(self.data)
        k2_stat, k2_p = stats.normaltest(self.data)
        
        # Visual checks
        plt.figure(figsize=(15, 5))
        
        # Histogram
        plt.subplot(131)
        sns.histplot(self.data, kde=True)
        plt.title('Distribution')
        
        # Q-Q plot
        plt.subplot(132)
        stats.probplot(self.data, dist="norm", plot=plt)
        plt.title('Q-Q Plot')
        
        # Box plot
        plt.subplot(133)
        sns.boxplot(y=self.data)
        plt.title('Box Plot')
        
        plt.tight_layout()
        plt.savefig('docs/4-stat-analysis/4.2-hypotheses-testing/assets/normality_check.png')
        plt.close()
        
        return {
            'shapiro_test': {'statistic': shapiro_stat, 'p_value': shapiro_p},
            'dagostino_test': {'statistic': k2_stat, 'p_value': k2_p},
            'interpretation': {
                'is_normal': shapiro_p > 0.05,
                'confidence': 'high' if min(shapiro_p, k2_p) > 0.1 else 'moderate'
            }
        }
```

### 2. Homogeneity of Variance 📊

```python
class VarianceChecker:
    """Toolkit for checking variance homogeneity"""
    
    def __init__(self, *groups):
        self.groups = groups
    
    def check_all(self):
        """Run comprehensive variance checks"""
        # Levene's test
        levene_stat, levene_p = stats.levene(*self.groups)
        
        # Bartlett's test
        bartlett_stat, bartlett_p = stats.bartlett(*self.groups)
        
        # Visual check
        plt.figure(figsize=(10, 5))
        
        # Box plots
        plt.subplot(121)
        sns.boxplot(data=self.groups)
        plt.title('Group Variances')
        
        # Spread-Location plot
        plt.subplot(122)
        means = [np.mean(group) for group in self.groups]
        vars = [np.var(group) for group in self.groups]
        plt.scatter(means, vars)
        plt.xlabel('Group Means')
        plt.ylabel('Group Variances')
        plt.title('Spread-Location Plot')
        
        plt.tight_layout()
        plt.savefig('docs/4-stat-analysis/4.2-hypotheses-testing/assets/variance_check.png')
        plt.close()
        
        return {
            'levene_test': {'statistic': levene_stat, 'p_value': levene_p},
            'bartlett_test': {'statistic': bartlett_stat, 'p_value': bartlett_p},
            'interpretation': {
                'homogeneous': levene_p > 0.05,
                'confidence': 'high' if min(levene_p, bartlett_p) > 0.1 else 'moderate'
            }
        }
```

## Effect Size Calculations 📏

```python
class EffectSizeCalculator:
    """Toolkit for calculating effect sizes"""
    
    def __init__(self, data1, data2=None):
        self.data1 = data1
        self.data2 = data2
    
    def calculate_all(self):
        """Calculate multiple effect size measures"""
        if self.data2 is None:
            # One-sample effect size
            d = np.mean(self.data1) / np.std(self.data1)
            return {
                'cohens_d': d,
                'interpretation': self._interpret_d(d)
            }
        else:
            # Two-sample effect sizes
            # Cohen's d
            d = (np.mean(self.data1) - np.mean(self.data2)) / \
                np.sqrt((np.var(self.data1) + np.var(self.data2)) / 2)
            
            # Glass's delta
            delta = (np.mean(self.data1) - np.mean(self.data2)) / np.std(self.data2)
            
            # Hedges' g
            n1, n2 = len(self.data1), len(self.data2)
            g = d * (1 - (3 / (4 * (n1 + n2) - 9)))
            
            return {
                'cohens_d': d,
                'glass_delta': delta,
                'hedges_g': g,
                'interpretation': self._interpret_d(d)
            }
    
    def _interpret_d(self, d):
        """Interpret Cohen's d effect size"""
        d = abs(d)
        if d < 0.2:
            return 'Very small effect'
        elif d < 0.5:
            return 'Small effect'
        elif d < 0.8:
            return 'Medium effect'
        else:
            return 'Large effect'
```

## Practice Questions 🤔
1. You're comparing customer satisfaction scores before and after a website redesign. Which test should you use and why?
2. A marketing team wants to know if purchase amounts vary by customer age group (18-25, 26-35, 36-50, 50+). What's the appropriate test?
3. How would you test if there's a relationship between email open rates and time of day?
4. Your A/B test shows p < 0.05 but a tiny effect size. What should you recommend?
5. When should you use non-parametric tests instead of their parametric counterparts?

## Key Takeaways 🎯
1. 📊 Choose tests based on data type and research question
2. ✅ Always check assumptions before testing
3. 📏 Consider effect sizes, not just p-values
4. 🔄 Use multiple approaches when appropriate
5. 📝 Document all decisions and interpretations

## Additional Resources 📚
- [Statistical Test Calculator](https://www.socscistatistics.com/)
- [Effect Size Calculator](https://www.psychometrica.de/effect_size.html)
- [Interactive Test Selection Guide](https://stats.idre.ucla.edu/other/mult-pkg/whatstat/)

Remember: Statistical tests are like tools in a toolbox - choose the right one for the job! 🛠️
