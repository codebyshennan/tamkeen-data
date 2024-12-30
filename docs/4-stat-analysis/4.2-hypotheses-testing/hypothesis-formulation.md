# Formulating Hypotheses: The Art of Scientific Questions 🔍

## Introduction: Why Hypotheses Matter 🎯
Think of a hypothesis as your scientific GPS - it guides your investigation and helps you arrive at meaningful conclusions. Whether you're testing a new drug, optimizing a website, or studying customer behavior, well-formulated hypotheses are your roadmap to discovery!

## The Anatomy of a Hypothesis 🔬

### The Dynamic Duo: Null and Alternative
1. **Null Hypothesis (H₀)** 🚫
   - The "nothing special happening" hypothesis
   - States no effect or no difference
   - What we try to disprove

2. **Alternative Hypothesis (H₁ or Hₐ)** ✨
   - The "something's happening" hypothesis
   - States there is an effect or difference
   - What we hope to support

\`\`\`python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def demonstrate_hypotheses():
    """
    Visualize the concept of null vs alternative hypotheses
    """
    # Generate data
    np.random.seed(42)
    
    # Null hypothesis data (no effect)
    null_data = np.random.normal(100, 15, 1000)
    
    # Alternative hypothesis data (with effect)
    alt_data = np.random.normal(105, 15, 1000)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    sns.histplot(null_data, label='Control', alpha=0.5)
    sns.histplot(alt_data, label='Treatment', alpha=0.5)
    plt.title('Data Distribution')
    plt.legend()
    
    plt.subplot(122)
    sns.boxplot(data=[null_data, alt_data])
    plt.xticks([0, 1], ['Control', 'Treatment'])
    plt.title('Group Comparison')
    
    plt.tight_layout()
    plt.savefig('docs/4-stat-analysis/4.2-hypotheses-testing/assets/hypothesis_visualization.png')
    plt.close()
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(null_data, alt_data)
    return t_stat, p_value

# Example usage
t_stat, p_value = demonstrate_hypotheses()
\`\`\`

## The Three Pillars of Good Hypotheses 🏛️

### 1. Specific and Clear 📝
Transform vague ideas into testable statements:

❌ Bad: "The treatment might work better"
✅ Good: "The new treatment reduces recovery time by at least 2 days"

\`\`\`python
def test_specific_hypothesis(control_data, treatment_data, min_improvement=2):
    """
    Test a specific hypothesis about treatment improvement
    
    H₀: treatment_effect ≤ min_improvement
    H₁: treatment_effect > min_improvement
    """
    # Calculate treatment effect
    effect = np.mean(control_data) - np.mean(treatment_data)
    
    # Calculate standard error
    n1, n2 = len(control_data), len(treatment_data)
    pooled_std = np.sqrt(((n1-1)*np.var(control_data) + (n2-1)*np.var(treatment_data)) / (n1+n2-2))
    se = pooled_std * np.sqrt(1/n1 + 1/n2)
    
    # Calculate test statistic
    t_stat = (effect - min_improvement) / se
    
    # One-tailed test
    p_value = 1 - stats.t.cdf(t_stat, df=n1+n2-2)
    
    return {
        'effect_size': effect,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
\`\`\`

### 2. Measurable 📊
Your hypothesis should involve quantifiable variables:

\`\`\`python
def measure_customer_satisfaction(ratings, target_score=4.0):
    """
    Analyze customer satisfaction metrics
    
    H₀: Mean satisfaction ≤ target_score
    H₁: Mean satisfaction > target_score
    """
    metrics = {
        'mean_score': np.mean(ratings),
        'median_score': np.median(ratings),
        'std_dev': np.std(ratings),
        'satisfaction_rate': np.mean(ratings >= 4),
        'sample_size': len(ratings)
    }
    
    # Statistical test
    t_stat, p_value = stats.ttest_1samp(ratings, target_score)
    
    # Create visualization
    plt.figure(figsize=(10, 5))
    
    plt.subplot(121)
    sns.histplot(ratings, bins=20)
    plt.axvline(target_score, color='r', linestyle='--', label=f'Target ({target_score})')
    plt.title('Distribution of Ratings')
    plt.legend()
    
    plt.subplot(122)
    sns.boxplot(y=ratings)
    plt.axhline(target_score, color='r', linestyle='--')
    plt.title('Rating Summary')
    
    plt.tight_layout()
    plt.savefig('docs/4-stat-analysis/4.2-hypotheses-testing/assets/satisfaction_analysis.png')
    plt.close()
    
    return {**metrics, 't_statistic': t_stat, 'p_value': p_value}
\`\`\`

### 3. Falsifiable ❌
Your hypothesis must be able to be proven wrong:

\`\`\`python
def demonstrate_falsifiability():
    """
    Show the importance of falsifiable hypotheses
    """
    # Example 1: Falsifiable hypothesis
    def test_mean_effect(data, threshold):
        """H₀: mean ≤ threshold"""
        t_stat, p_value = stats.ttest_1samp(data, threshold)
        return p_value < 0.05
    
    # Example 2: Non-falsifiable statement
    def vague_statement(data):
        """'The treatment might help some people'"""
        return "Statement too vague to test statistically"
    
    # Demonstrate with simulated data
    np.random.seed(42)
    data = np.random.normal(loc=10, scale=2, size=100)
    
    return {
        'falsifiable_result': test_mean_effect(data, 9.5),
        'non_falsifiable': vague_statement(data)
    }
\`\`\`

## Types of Hypotheses 📚

### 1. Simple vs Composite Hypotheses
- Simple: Tests exact value
- Composite: Tests range of values

\`\`\`python
def demonstrate_hypothesis_types(data):
    """Compare simple and composite hypotheses"""
    # Simple hypothesis (H₀: μ = 100)
    simple_test = stats.ttest_1samp(data, 100)
    
    # Composite hypothesis (H₀: 95 ≤ μ ≤ 105)
    mean = np.mean(data)
    composite_result = 95 <= mean <= 105
    
    return {
        'simple_p_value': simple_test.pvalue,
        'composite_result': composite_result
    }
\`\`\`

### 2. Directional vs Non-directional 🔄
\`\`\`python
def compare_directional_tests(control, treatment):
    """
    Compare one-tailed and two-tailed tests
    
    One-tailed: H₁: treatment > control
    Two-tailed: H₁: treatment ≠ control
    """
    # Two-tailed test
    two_tail = stats.ttest_ind(treatment, control)
    
    # One-tailed test
    one_tail = stats.ttest_ind(treatment, control, alternative='greater')
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    sns.histplot(control, label='Control', alpha=0.5)
    sns.histplot(treatment, label='Treatment', alpha=0.5)
    plt.title('Distribution Comparison')
    plt.legend()
    
    plt.subplot(122)
    plt.text(0.1, 0.7, f"Two-tailed p-value: {two_tail.pvalue:.4f}")
    plt.text(0.1, 0.5, f"One-tailed p-value: {one_tail.pvalue:.4f}")
    plt.axis('off')
    plt.title('Test Results')
    
    plt.tight_layout()
    plt.savefig('docs/4-stat-analysis/4.2-hypotheses-testing/assets/directional_tests.png')
    plt.close()
    
    return {
        'two_tailed': two_tail,
        'one_tailed': one_tail
    }
\`\`\`

## Common Pitfalls and Solutions ⚠️

### 1. Multiple Testing Problem 🎯
When testing multiple hypotheses, adjust for multiple comparisons:

\`\`\`python
def handle_multiple_testing(p_values):
    """
    Apply corrections for multiple testing
    """
    from statsmodels.stats.multitest import multipletests
    
    # Different correction methods
    corrections = {
        'bonferroni': multipletests(p_values, method='bonferroni')[1],
        'fdr': multipletests(p_values, method='fdr_bh')[1],
        'holm': multipletests(p_values, method='holm')[1]
    }
    
    # Visualize
    plt.figure(figsize=(10, 6))
    
    for i, (method, values) in enumerate(corrections.items()):
        plt.subplot(1, 3, i+1)
        plt.scatter(p_values, values)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.title(f'{method.capitalize()} Correction')
        plt.xlabel('Original p-value')
        plt.ylabel('Adjusted p-value')
    
    plt.tight_layout()
    plt.savefig('docs/4-stat-analysis/4.2-hypotheses-testing/assets/multiple_testing.png')
    plt.close()
    
    return corrections
\`\`\`

### 2. P-hacking 🎲
Don't fish for significance:

\`\`\`python
def demonstrate_p_hacking(data):
    """Show why p-hacking is problematic"""
    results = []
    
    # DON'T do this!
    for cutoff in range(10, len(data), 10):
        subset = data[:cutoff]
        _, p_value = stats.ttest_1samp(subset, 0)
        results.append({
            'sample_size': cutoff,
            'p_value': p_value
        })
    
    # Visualize the problem
    plt.figure(figsize=(8, 5))
    df = pd.DataFrame(results)
    plt.plot(df['sample_size'], df['p_value'])
    plt.axhline(0.05, color='r', linestyle='--', label='Significance Level')
    plt.title('P-hacking Demonstration')
    plt.xlabel('Sample Size')
    plt.ylabel('P-value')
    plt.legend()
    plt.savefig('docs/4-stat-analysis/4.2-hypotheses-testing/assets/p_hacking.png')
    plt.close()
    
    return df
\`\`\`

## Best Practices for Success 🌟

### 1. Plan Your Analysis 📋
```python
class HypothesisTest:
    def __init__(self, name, null_hypothesis, alternative_hypothesis):
        self.name = name
        self.null = null_hypothesis
        self.alternative = alternative_hypothesis
        self.data = {}
        self.results = None
        
    def add_data(self, group_name, data):
        self.data[group_name] = data
        
    def run_test(self):
        # Implement specific test logic
        pass
    
    def visualize(self):
        # Create visualizations
        pass
    
    def report(self):
        # Generate summary report
        pass
```

### 2. Document Everything 📝
- State hypotheses clearly
- Specify significance level
- Record all decisions
- Note any deviations

### 3. Consider Practical Significance 💭
Statistical significance ≠ Practical importance

## Practice Questions 🤔
1. Write null and alternative hypotheses for testing if a new teaching method improves test scores.
2. When would you use a one-tailed vs two-tailed test? Give examples.
3. How would you handle testing multiple features of a website simultaneously?
4. Design a hypothesis test for comparing customer satisfaction across three different stores.
5. What's wrong with p-hacking and how can you avoid it?

## Key Takeaways 🎯
1. 📝 Clear, specific hypotheses guide good research
2. 📊 Make your hypotheses measurable and falsifiable
3. ⚠️ Beware of multiple testing and p-hacking
4. 🔍 Consider both statistical and practical significance
5. 📈 Document your decisions and rationale

## Additional Resources 📚
- [Statistical Hypothesis Testing Guide](https://www.statisticshowto.com/probability-and-statistics/hypothesis-testing/)
- [Multiple Testing Calculator](https://www.statstest.com/bonferroni/)
- [P-value Misconceptions](https://www.nature.com/articles/nmeth.3288)

Remember: A well-formulated hypothesis is half the battle won! 🎯
