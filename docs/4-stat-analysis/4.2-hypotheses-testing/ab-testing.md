---
reading_minutes: 40
objectives:
  - Calculate per-arm sample size from baseline rate, MDE, alpha, and power before the test starts.
  - Run a sample-ratio check to detect broken randomization before interpreting results.
  - Decide whether to ship using statistical significance, the CI on the lift, and a minimum practical effect threshold.
  - Avoid peeking, novelty effects, and t-tests on proportional outcomes.
---

# A/B Testing: Making Data-Driven Decisions

**After this lesson:** you can explain the core ideas in “A/B Testing: Making Data-Driven Decisions” and reproduce the examples here in your own notebook or environment.

## Overview

A/B testing is the product-facing form of a two-arm experiment: **control** vs **treatment**, a clear **metric**, random assignment when possible, and rules for how long to run and how to read uncertainty. You already know how to phrase hypotheses and pick tests; here the focus is workflow, metrics, and interpretation in a live or simulated environment.

## Why this matters

- You will run **controlled comparisons** of two variants and interpret results without fooling yourself.
- You will link randomization, metrics, and stopping rules to business or product decisions.

## Prerequisites

- [Formulating hypotheses](./hypothesis-formulation.md) and [Statistical tests](./statistical-tests.md) (how tests connect to metrics and designs).
- [Experimental design](./experimental-design.md) for control and randomization.

> **Note:** Pair this lesson with the [tutorial notebook](./hypothesis-testing.ipynb) if your cohort uses it.

## Introduction: Why A/B Testing?

Imagine you're a chef trying to improve a recipe. Instead of guessing what changes might work better, you could serve two versions and see which one customers prefer. That's A/B testing in a nutshell - a scientific way to compare options and make data-driven decisions!

## The Basics: Control vs Treatment

{% include mermaid-diagram.html src="4-stat-analysis/4.2-hypotheses-testing/diagrams/ab-testing-1.mmd" %}

### What is A/B Testing?

A/B testing (or split testing) is like running a scientific experiment where you:

1. Take two versions of something (A and B)
2. Show them randomly to different groups
3. Measure which performs better

### Common Applications

- Website optimization
- Email marketing campaigns
- Mobile app features
- Pricing strategies
- UI/UX design choices

**`ABTest` helper: summaries, CIs, and a three-panel figure**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class ABTest:
    """
    A comprehensive A/B testing framework
    """
    def __init__(self, control_data, treatment_data, metric_name="conversion"):
        self.control = control_data
        self.treatment = treatment_data
        self.metric_name = metric_name
        
        # Calculate basic statistics
        self.control_stats = self._calculate_stats(control_data)
        self.treatment_stats = self._calculate_stats(treatment_data)
        
    def _calculate_stats(self, data):
        """Calculate key statistics for a group"""
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'size': len(data),
            'ci': stats.t.interval(
                0.95, 
                len(data)-1,
                loc=np.mean(data),
                scale=stats.sem(data)
            )
        }
    
    def visualize(self):
        """Create comprehensive visualization of results"""
        plt.figure(figsize=(15, 5))
        
        # Distribution comparison
        plt.subplot(131)
        sns.kdeplot(self.control, label='Control (A)', shade=True)
        sns.kdeplot(self.treatment, label='Treatment (B)', shade=True)
        plt.title('Distribution Comparison')
        plt.xlabel(self.metric_name)
        plt.legend()
        
        # Box plot
        plt.subplot(132)
        sns.boxplot(data=[self.control, self.treatment])
        plt.xticks([0, 1], ['Control (A)', 'Treatment (B)'])
        plt.title('Box Plot Comparison')
        
        # Effect size visualization
        plt.subplot(133)
        effect_size = (np.mean(self.treatment) - np.mean(self.control)) / \
                     np.std(self.control)
        plt.bar(['Effect Size'], [effect_size])
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Standardized Effect Size')
        
        plt.tight_layout()
        plt.show()

        return self
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-10" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports and class definition</span>
    </div>
    <div class="code-callout__body">
      <p>Import NumPy, pandas, scipy stats, Matplotlib, and Seaborn — the standard stack for statistical analysis and visualization. The <code>ABTest</code> class wraps both arms of the experiment and the methods that operate on them, keeping the analysis self-contained.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="11-18" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Store both arms, compute stats immediately</span>
    </div>
    <div class="code-callout__body">
      <p>The constructor stores control and treatment data, then immediately computes summary statistics for both. Doing this in <code>__init__</code> means every method can access <code>self.control_stats</code> without recomputing.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="20-32" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">95% confidence interval</span>
    </div>
    <div class="code-callout__body">
      <p><code>stats.t.interval(0.95, ...)</code> returns a (lower, upper) interval using the t-distribution with <code>n-1</code> degrees of freedom. <code>stats.sem</code> computes the standard error of the mean — the CI shows the range where the true mean likely falls.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="34-50" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Distribution &amp; box plot views</span>
    </div>
    <div class="code-callout__body">
      <p>KDE plots reveal distributional shape (skew, bimodality) that summary stats hide. The box plot shows median, IQR, and outliers side by side — together they give a fuller picture than just comparing means.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="52-64" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Effect size, save, and return</span>
    </div>
    <div class="code-callout__body">
      <p>(Treatment mean − Control mean) ÷ Control std is a Cohen's d–style effect size. Unlike a p-value, it tells you <em>how large</em> the difference is in standard deviation units. Returning <code>self</code> allows method chaining.</p>
    </div>
  </div>
</aside>
</div>

## Setting Up Your A/B Test

### 1. Sample Size Calculation

Don't start without knowing how many samples you need!

**Approximate per-arm sample size (normal approximation)**

```python
def calculate_sample_size(
    baseline_rate=0.1,    # Current conversion rate
    mde=0.02,            # Minimum detectable effect
    alpha=0.05,          # Significance level
    power=0.8            # Statistical power
):
    """
    Calculate required sample size for A/B test
    
    Parameters:
    -----------
    baseline_rate : float
        Current conversion rate (e.g., 0.10 for 10%)
    mde : float
        Minimum detectable effect (e.g., 0.02 for 2% increase)
    alpha : float
        Significance level (Type I error rate)
    power : float
        Statistical power (1 - Type II error rate)
        
    Returns:
    --------
    dict with required sample sizes and test parameters
    """
    # Standard normal critical values
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Calculate pooled standard deviation
    p_pooled = baseline_rate + (baseline_rate * mde/2)
    
    # Calculate sample size
    n = np.ceil(
        (2 * p_pooled * (1 - p_pooled) * (z_alpha + z_beta)**2) / mde**2
    )
    
    return {
        'sample_size_per_group': int(n),
        'total_sample_size': int(2 * n),
        'parameters': {
            'baseline_rate': baseline_rate,
            'mde': mde,
            'alpha': alpha,
            'power': power
        }
    }

# Example usage
sample_size_calc = calculate_sample_size(
    baseline_rate=0.10,  # 10% current conversion
    mde=0.02,           # Want to detect 2% improvement
    alpha=0.05,         # 5% significance level
    power=0.8           # 80% power
)
```

### 2. Random Assignment

Ensure fair comparison with proper randomization:

**Shuffle-and-slice assignment**

```python
def assign_to_groups(user_ids, split_ratio=0.5, seed=None):
    """
    Randomly assign users to control and treatment groups
    
    Parameters:
    -----------
    user_ids : array-like
        List of user IDs to assign
    split_ratio : float
        Proportion to assign to treatment (default: 0.5)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    dict with group assignments
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Shuffle user IDs
    user_ids = np.array(user_ids)
    np.random.shuffle(user_ids)
    
    # Split into groups
    split_point = int(len(user_ids) * split_ratio)
    
    assignments = {
        'control': user_ids[:split_point],
        'treatment': user_ids[split_point:]
    }
    
    # Verify split
    print(f"Control group size: {len(assignments['control'])}")
    print(f"Treatment group size: {len(assignments['treatment'])}")
    print(f"Actual split ratio: {len(assignments['treatment'])/len(user_ids):.2%}")
    
    return assignments
```

## Running Your Test

### 1. Data Collection

Track everything systematically:

**Row-level event log → `DataFrame` aggregation**

```python
class ABTestDataCollector:
    def __init__(self):
        self.data = []
        self.start_time = pd.Timestamp.now()
        
    def record_observation(self, user_id, group, metric_value, metadata=None):
        """Record a single observation"""
        observation = {
            'user_id': user_id,
            'group': group,
            'value': metric_value,
            'timestamp': pd.Timestamp.now(),
            'days_in_test': (pd.Timestamp.now() - self.start_time).days
        }
        
        if metadata:
            observation.update(metadata)
            
        self.data.append(observation)
    
    def get_results(self):
        """Get current test results"""
        df = pd.DataFrame(self.data)
        
        # Calculate key metrics by group
        results = df.groupby('group').agg({
            'value': ['count', 'mean', 'std'],
            'user_id': 'nunique'
        })
        
        return results
```

### 2. Monitoring

Watch your test without peeking too much:

**Interim dashboard: sample size gate + running p-value**

```python
def monitor_test(data_collector, min_sample_size):
    """
    Monitor ongoing A/B test
    
    Parameters:
    -----------
    data_collector : ABTestDataCollector
        Object containing test data
    min_sample_size : int
        Minimum required sample size
        
    Returns:
    --------
    dict with monitoring metrics
    """
    results = data_collector.get_results()
    
    # Check if we've reached minimum sample size
    current_size = results.loc[:, ('value', 'count')].min()
    size_reached = current_size >= min_sample_size
    
    # Calculate current p-value
    control_data = [x['value'] for x in data_collector.data 
                   if x['group'] == 'control']
    treatment_data = [x['value'] for x in data_collector.data 
                     if x['group'] == 'treatment']
    
    _, p_value = stats.ttest_ind(treatment_data, control_data)
    
    return {
        'current_size': current_size,
        'target_size': min_sample_size,
        'size_reached': size_reached,
        'p_value': p_value,
        'can_conclude': size_reached and p_value < 0.05
    }
```

## Analysis and Decision Making

### 1. Statistical Analysis

Don't just look at the numbers - understand them:

**Means, relative lift, t-test, and a CI on the mean difference**

```python
def analyze_results(control_data, treatment_data, alpha=0.05):
    """
    Comprehensive A/B test analysis
    
    Returns:
    --------
    dict with test results and recommendations
    """
    # Basic statistics
    control_mean = np.mean(control_data)
    treatment_mean = np.mean(treatment_data)
    
    # Effect size (relative change)
    relative_change = (treatment_mean - control_mean) / control_mean
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(treatment_data, control_data)
    
    # Confidence interval for difference
    ci = stats.t.interval(
        1 - alpha,
        len(control_data) + len(treatment_data) - 2,
        loc=treatment_mean - control_mean,
        scale=np.sqrt(np.var(treatment_data)/len(treatment_data) + 
                     np.var(control_data)/len(control_data))
    )
    
    return {
        'metrics': {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'relative_change': relative_change,
            'absolute_change': treatment_mean - control_mean
        },
        'statistical_tests': {
            't_statistic': t_stat,
            'p_value': p_value,
            'confidence_interval': ci
        },
        'recommendation': 'accept' if p_value < alpha and relative_change > 0 
                         else 'reject'
    }
```

### 2. Making the Call

Statistical significance alone doesn't justify shipping. Use a two-axis check:

| | Stat. significant | Not stat. significant |
|---|---|---|
| **Practically significant** | Ship it | Underpowered — re-run with more data |
| **Not practically significant** | Don't ship — real but trivial | Don't ship |

**Step 1: Set a minimum practical effect upfront.** Before the test starts, define the smallest effect worth shipping (e.g., "≥ 2% lift in conversion"). Without this, any positive result feels meaningful.

**Step 2: Read the confidence interval, not just the p-value.** A 95% CI of [0.001%, 3.9%] straddles a 2% threshold — you can't confidently claim the treatment meets your bar, even if p < 0.05.

**Step 3: Factor in implementation costs.** A statistically and practically significant result still has an engineering and ops cost. Estimate the expected value before committing:

```python
def expected_value(lift, baseline_rate, daily_users, revenue_per_conversion, implementation_cost):
    """
    Estimate 30-day expected value of shipping treatment.

    lift: relative improvement (e.g. 0.02 for 2%)
    """
    daily_conversions_gained = daily_users * baseline_rate * lift
    monthly_revenue_gain = daily_conversions_gained * revenue_per_conversion * 30
    return monthly_revenue_gain - implementation_cost

# Example: 2% lift, 10% baseline, 10k users/day, $5 per conversion, $20k implementation
ev = expected_value(0.02, 0.10, 10_000, 5, 20_000)
print(f"Expected 30-day value: ${ev:,.0f}")  # Expected 30-day value: $10,000
```

```
Expected 30-day value: $-17,000
```

**Step 4: Document the decision.** Record what you shipped and why — future tests on the same surface need this context to avoid re-testing the same thing.

**Decision checklist:**
- [ ] p-value < α (statistical significance confirmed)
- [ ] Effect size ≥ minimum practical threshold
- [ ] CI lower bound above zero (or your MDE)
- [ ] Sample ratio check passed (`check_sample_ratio` returned `is_valid: True`)
- [ ] No external events during the test window that could confound results
- [ ] Cost-benefit analysis positive

## Common Pitfalls and Solutions

### 1. Peeking Problem

Don't keep checking results - it increases false positives!

**Bonferroni-style peeking adjustment (illustrative)**

```python
def adjust_for_peeking(p_values, total_looks):
    """Adjust significance level for multiple looks at the data"""
    from statsmodels.stats.multitest import multipletests
    
    # Use Bonferroni correction
    alpha_per_peek = 0.05 / total_looks
    
    # Adjust p-values
    rejected, adjusted_p_values, _, _ = multipletests(
        p_values, 
        alpha=alpha_per_peek, 
        method='bonferroni'
    )
    
    return {
        'original_p_values': p_values,
        'adjusted_p_values': adjusted_p_values,
        'significant': rejected
    }
```


### 2. Sample Ratio Mismatch

Check if your randomization is working:

**Chi-square goodness-of-fit on assignment counts**

```python
def check_sample_ratio(control_size, treatment_size, expected_ratio=0.5):
    """
    Check if sample ratio matches expected split
    
    Returns:
    --------
    dict with ratio analysis results
    """
    total = control_size + treatment_size
    actual_ratio = treatment_size / total
    
    # Chi-square test for ratio
    expected = [total * (1-expected_ratio), total * expected_ratio]
    observed = [control_size, treatment_size]
    
    _, p_value = stats.chisquare(observed, expected)
    
    return {
        'expected_ratio': expected_ratio,
        'actual_ratio': actual_ratio,
        'p_value': p_value,
        'is_valid': p_value > 0.05,
        'recommendation': 'valid split' if p_value > 0.05 else 'investigate split'
    }
```


## Gotchas

- **Peeking at results and stopping early** — checking the p-value repeatedly as data accumulates inflates the false-positive rate well above your stated α. The lesson's `monitor_test` helper naively ANDs size and p-value; in production, use sequential methods (e.g., alpha spending functions or Bayesian stopping rules) instead of a plain `p < 0.05` gate.
- **Calculating sample size after collecting data** — the `calculate_sample_size` function must be called *before* the experiment starts. Running it post-hoc and adjusting the target to match what you collected is p-hacking masquerading as power analysis.
- **Sample ratio mismatch going undetected** — if your randomization mechanism is broken (e.g., a caching layer serves one variant more often), your observed groups will be unequal in ways that invalidate the independence assumption. Always run `check_sample_ratio` before interpreting results; a chi-square p-value below 0.05 on the assignment counts is a red flag.
- **Using a t-test when your metric is a proportion** — the lesson's `analyze_results` function calls `ttest_ind` on raw arrays. For binary conversion outcomes (0/1), a two-proportion z-test or chi-square test on the contingency table (as shown in the A/B chi-square example) is the more principled choice, especially for small or imbalanced samples.
- **Interpreting the `recommendation: 'accept'` flag as a deployment decision** — the flag in `analyze_results` fires when `p_value < alpha and relative_change > 0`. A positive relative change of 0.01% is statistically significant with enough traffic but may not justify the engineering cost; always pair the flag with a minimum practical effect threshold agreed on before the test.
- **Ignoring novelty effect and network effects** — users interacting with a new variant immediately after launch may behave differently than they will once the novelty wears off, and in social or referral products, treatment users may influence control users. Both effects bias effect-size estimates and are invisible to standard two-arm analysis.

## Next steps

- Continue to [Results analysis](./results-analysis.md).

## Additional Resources

- [A/B Testing Calculator](https://www.abtestguide.com/calc/)
- [Sample Size Calculator](https://www.optimizely.com/sample-size-calculator/)
- [A/B Testing Best Practices](https://www.optimizely.com/optimization-glossary/ab-testing/)

Remember: A/B testing is a powerful tool, but only when used correctly!

