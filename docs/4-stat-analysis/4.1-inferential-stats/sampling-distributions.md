---
reading_minutes: 30
objectives:
  - Explain what a sampling distribution is and how it differs from a sample distribution.
  - State the Central Limit Theorem and judge when it applies given a population shape.
  - Compute the standard error for a mean, a proportion, and a difference of means.
  - Use the σ/√n relationship to predict how precision scales with sample size.
---

# Sampling Distributions: The Heart of Statistical Inference

**After this lesson:** you can explain Sampling Distributions: The Heart of Statistical Inference and try the examples in your own notebook.

## Overview

If you drew another sample tomorrow, your mean would change slightly. A **sampling distribution** describes how a statistic (like \\(\bar x\\)) would vary under repeated sampling. That distribution is what makes [confidence intervals](./confidence-intervals.md), p-values, and tests behave the way they do, including the famous **Central Limit Theorem** for means. Read this lesson before confidence intervals so the margin-of-error formula has a foundation.

## Why this matters

- **Sampling distributions** explain why means and proportions vary from sample to sample.
- The **Central Limit Theorem** and **standard error** underpin confidence intervals and tests.

## Prerequisites

- [Population vs sample](./population-sample.md) for the population/sample/parameter/statistic vocabulary.
- Comfort with means, standard deviation, and basic probability.

> **Note:** Simulation plots in this lesson are optional; the written CLT summary is the core outcome.

## Introduction: Why Sampling Distributions Matter

Imagine you're a chef trying to perfect a recipe. You taste-test small portions (samples) to understand how the entire dish (population) tastes. But how reliable are these taste tests? That's where sampling distributions come in - they help us understand how sample statistics vary and how well they represent the true population!

![Sampling Distribution Concept](assets/sampling_distribution_comparison.png)
*Figure 1: Visual representation of sampling distributions. The diagram shows how multiple samples from a population create a distribution of sample statistics.*

## What is a Sampling Distribution?

A sampling distribution is the distribution of a statistic (like mean or proportion) calculated from repeated random samples of the same size from a population. Think of it as the "distribution of distributions" - it shows us how sample statistics bounce around the true population value.

### Mathematical Definition

For a sample mean \\(\bar x\\):

- Mean: \\(E(\bar x) = \mu\\) (population mean)
- Standard Error: \\(SE(\bar x) = \dfrac{\sigma}{\sqrt{n}}\\)
  - where \\(\sigma\\) is the population standard deviation
  - and \\(n\\) is the sample size

![Sampling Distribution Formula](assets/standard_error_visualization.png)
*Figure 2: Visual explanation of the sampling distribution formula. The diagram shows how the standard error decreases as sample size increases.*

## The Central Limit Theorem (CLT): Statistical Magic

### What is CLT?

The Central Limit Theorem states that for sufficiently large samples:

1. The sampling distribution of the mean is approximately normal
2. This holds true regardless of the population's distribution
3. The larger the sample size, the more normal it becomes

#### How large is "sufficiently large"?

The folklore answer is "n ≥ 30," which is fine as a default but hides important nuance: how fast the sampling distribution becomes approximately normal depends on the *shape* of the underlying population. The mini-pictures below show the rough shape implied by each row.

| Shape | Population shape | Approximate \\(n\\) needed |
|---|---|---|
| ![Normal](assets/shape_normal.png) | Already normal (bell curve) | any \\(n\\), the sampling distribution of \\(\bar x\\) is exactly normal |
| ![Uniform](assets/shape_uniform.png) | Roughly symmetric with light tails (e.g., uniform / "flat top") | \\(n \approx 15\text{-}20\\) |
| ![Mild skew](assets/shape_mild_skew.png) | Mild skew or moderate outliers (gentle lean to one side) | \\(n \approx 30\text{-}40\\), the textbook rule |
| ![Strong skew](assets/shape_strong_skew.png) | Strong skew (long tail, e.g., exponential, income, waiting times) | \\(n \approx 50\text{-}100\\) or more |
| ![Binomial extreme](assets/shape_binomial_extreme.png) | Binomial with \\(p\\) near 0 or 1 (most weight on one side) | check \\(np \geq 10\\) **and** \\(n(1-p) \geq 10\\) |

**How to read this:** the bumpier or more lop-sided the population, the more data you need before the *average* of a sample starts to look like a clean bell. So when you read "n = 30 is enough," ask: *enough for what kind of population?* If your data have a long tail or many extreme values (think incomes, waiting times, customer spend), aim for the higher end of the table.

Look at it in action!

**CLT simulation: population → one sample → distribution of \\(\bar x\\)**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def demonstrate_clt(distribution='exponential', sample_size=30, n_samples=1000):
    """
    Demonstrate CLT with different distributions
    """
    plt.figure(figsize=(15, 5))

    # Generate population
    if distribution == 'exponential':
        population = np.random.exponential(scale=1.0, size=10000)
        title = 'Exponential Distribution'
    elif distribution == 'uniform':
        population = np.random.uniform(0, 1, 10000)
        title = 'Uniform Distribution'
    else:  # Skewed custom distribution
        population = np.concatenate([
            np.random.normal(0, 1, 7000),
            np.random.normal(3, 0.5, 3000)
        ])
        title = 'Skewed Distribution'

    # Take many samples and calculate their means
    sample_means = [
        np.mean(np.random.choice(population, size=sample_size))
        for _ in range(n_samples)
    ]

    # Plot results
    plt.subplot(131)
    plt.hist(population, bins=50, density=True, alpha=0.7, color='skyblue')
    plt.title(f'Population Distribution\n({title})')
    plt.xlabel('Value')
    plt.ylabel('Density')

    plt.subplot(132)
    sample = np.random.choice(population, size=sample_size)
    plt.hist(sample, bins=20, density=True, alpha=0.7, color='lightgreen')
    plt.title(f'One Sample Distribution\n(n={sample_size})')
    plt.xlabel('Value')

    plt.subplot(133)
    plt.hist(sample_means, bins=30, density=True, alpha=0.7, color='salmon')
    x = np.linspace(min(sample_means), max(sample_means), 100)
    plt.plot(x, stats.norm.pdf(x, np.mean(sample_means), np.std(sample_means)),
             'k--', label='Normal Curve')
    plt.title(f'Sampling Distribution\nof the Mean')
    plt.xlabel('Sample Mean')
    plt.legend()

    plt.tight_layout()
    return plt

# Create and save plots for different distributions
distributions = ['exponential', 'uniform', 'skewed']
for dist in distributions:
    plt = demonstrate_clt(distribution=dist)
    plt.savefig(f'docs/4-stat-analysis/4.1-inferential-stats/assets/clt_{dist}.png')
    plt.close()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="10-23" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Build population</span>
    </div>
    <div class="code-callout__body">
      <p>Switch on <code>distribution</code> to build an exponential, uniform, or bimodal skewed population of 10,000 values.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="26-29" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Collect sample means</span>
    </div>
    <div class="code-callout__body">
      <p>Repeatedly sample n=30 from the population and store each mean. This list is the empirical sampling distribution.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="46-50" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Overlay the normal curve</span>
    </div>
    <div class="code-callout__body">
      <p>Fit a normal curve to the simulated means and overlay it on the histogram, the visual signature of CLT.</p>
    </div>
  </div>
</aside>
</div>

### What the simulation produces

Three side-by-side panels per population type:

1. **Population**: the raw shape we sampled from (could be very non-normal).
2. **One sample**: what 30 randomly drawn values look like (still messy).
3. **Sampling distribution**: histogram of *means* from 1,000 such samples, with a normal curve overlaid.

Look at panel 3 in each image: even for the lopsided exponential and bimodal "skewed" populations, the histogram of means is a clean bell.

![CLT, exponential population](assets/clt_exponential.png)
*Figure 2a: An exponential (long-tail) population. Individual draws are skewed, but the sampling distribution of the mean (right) is approximately normal.*

![CLT, uniform population](assets/clt_uniform.png)
*Figure 2b: A uniform (flat) population. Same story, the mean's distribution looks like a bell even though the population doesn't.*

![CLT, bimodal/skewed population](assets/clt_skewed.png)
*Figure 2c: A bumpy, two-bump population. CLT still kicks in for the mean.*

**Big idea:** CLT is a statement about *averages*, not about individual data points. The raw data can stay weird; the mean smooths out.

### Interactive simulation: try it yourself

Use the **dropdown** to switch the population shape (normal, uniform, exponential, bimodal) and the **slider** to change the sample size \\(n\\). Watch the right-hand panel: even when the population (left) is wildly non-normal, the histogram of sample means becomes increasingly bell-shaped as \\(n\\) grows.

<iframe src="assets/interactive/clt_simulation.html" width="100%" height="500" frameborder="0" loading="lazy" title="Interactive CLT simulation"></iframe>

**Things to try:**

- Set the population to **right-skew (exponential)** with \\(n = 5\\). Notice the sampling distribution is still skewed.
- Bump \\(n\\) to 30, then 100. The right panel becomes a textbook bell.
- Switch to **bimodal**. Even though the population has two humps, the *means* land in a single bell. That's CLT.

## Standard Error: Measuring the Spread

The standard error (SE) tells us how much sample statistics typically deviate from the population parameter. It's like a "margin of wobble" for our estimates!

### Formula for Different Statistics

1. For means: \\(SE(\bar x) = \dfrac{\sigma}{\sqrt{n}}\\)
2. For proportions: \\(SE(\hat p) = \sqrt{\dfrac{p(1-p)}{n}}\\)
3. For differences: \\(SE(\bar x_1 - \bar x_2) = \sqrt{\dfrac{\sigma_1^2}{n_1} + \dfrac{\sigma_2^2}{n_2}}\\)

#### Symbol legend

| Symbol | Read aloud | What it means |
|---|---|---|
| \\(SE\\) | "standard error" | How much the statistic typically wobbles from sample to sample |
| \\(\bar x\\) | "x-bar" | The **sample mean** (one number computed from your data) |
| \\(\sigma\\) | "sigma" | The **population standard deviation**, the spread of individual values across the whole population |
| \\(\sigma^2\\) | "sigma squared" | The **population variance** (just \\(\sigma\\) squared) |
| \\(n\\) | "n" | The **sample size**, how many observations you collected |
| \\(\sqrt{n}\\) | "square root of n" | Why bigger samples shrink the SE more slowly than you'd hope (you need 4× more data to halve the SE) |
| \\(\hat p\\) | "p-hat" | The **sample proportion**, the fraction of "yes" outcomes in your sample (e.g., 184 / 200 = 0.92) |
| \\(p\\) | "p" | The **true population proportion** that \\(\hat p\\) is estimating |
| \\(p(1-p)\\) | "p times one minus p" | A measure of how "spread out" a yes/no variable is; biggest at \\(p = 0.5\\), smallest near 0 or 1 |
| \\(\bar x_1 - \bar x_2\\) | "x-bar one minus x-bar two" | The **difference between two group means** (e.g., treatment minus control) |
| \\(n_1, n_2\\) | "n one, n two" | The sample sizes of group 1 and group 2 |
| \\(\sigma_1, \sigma_2\\) | "sigma one, sigma two" | The population standard deviations of group 1 and group 2 |

In plain words, all three formulas have the same shape: **noise in the data ÷ a function of how many points you have**. More data → smaller SE → tighter estimates.

Look at how sample size affects SE:

**Empirical vs theoretical standard error across n**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def demonstrate_standard_error():
    """
    Show how SE changes with sample size
    """
    # Generate population
    np.random.seed(42)
    population = np.random.normal(100, 15, 10000)

    # Test different sample sizes
    sizes = [10, 30, 100, 300, 1000]
    results = []

    # Visualize the effect of sample size on SE
    plt.figure(figsize=(12, 8))
    for i, n in enumerate(sizes):
        plt.subplot(2, 3, i+1)

        # Take multiple samples
        sample_means = [
            np.mean(np.random.choice(population, size=n))
            for _ in range(1000)
        ]

        # Plot sampling distribution
        plt.hist(sample_means, bins=30, density=True, alpha=0.7)
        plt.axvline(np.mean(population), color='red', linestyle='--',
                   label='Population Mean')

        # Add normal curve
        x = np.linspace(min(sample_means), max(sample_means), 100)
        plt.plot(x, stats.norm.pdf(x, np.mean(sample_means), np.std(sample_means)),
                'k--', label='Normal Curve')

        plt.title(f'Sample Size: {n}\nSE: {np.std(sample_means):.2f}')
        plt.legend()

    plt.tight_layout()
    plt.savefig('docs/4-stat-analysis/4.1-inferential-stats/assets/standard_error_effect.png')
    plt.close()

    # Calculate theoretical and empirical SE
    for n in sizes:
        # Theoretical SE
        theoretical_se = np.std(population) / np.sqrt(n)

        # Empirical SE (from sampling distribution)
        sample_means = [
            np.mean(np.random.choice(population, size=n))
            for _ in range(1000)
        ]
        empirical_se = np.std(sample_means)

        results.append({
            'size': n,
            'theoretical': theoretical_se,
            'empirical': empirical_se
        })

    return results

# Run demonstration
se_results = demonstrate_standard_error()
print("\nStandard Error Analysis")
print("Sample Size | Theoretical SE | Empirical SE")
print("-" * 45)
for r in se_results:
    print(f"{r['size']:^10d} | {r['theoretical']:^13.3f} | {r['empirical']:^11.3f}")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="5-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Population setup</span>
    </div>
    <div class="code-callout__body">
      <p>Create a synthetic normal population of 10,000 values with mean 100 and SD 15 as the reference distribution.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-36" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Subplot grid</span>
    </div>
    <div class="code-callout__body">
      <p>For each n, simulate 1,000 sample means, plot the sampling distribution as a histogram with a normal curve overlay and the SE in the title.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="41-56" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Theoretical vs empirical</span>
    </div>
    <div class="code-callout__body">
      <p>Compute both the formula-based SE (σ/√n) and the Monte Carlo SE for each sample size and store them for the table printout.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="60-65" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Comparison table</span>
    </div>
    <div class="code-callout__body">
      <p>Print a formatted table comparing theoretical and empirical SE values so students can verify the formula against simulation.</p>
    </div>
  </div>
</aside>
</div>

![Standard error effect](assets/standard_error_effect.png)
*Figure 3: As sample size grows from 10 to 1,000, the sampling distribution of the mean (each panel) gets narrower, the standard error shrinks roughly by 1/√n.*

#### Interactive: SE vs sample size

Slide \\(n\\) from 5 to 500. Left: the sampling distribution of \\(\bar x\\) gets narrower. Right: the empirical SE (orange dot) lands close to the theoretical \\(\sigma/\sqrt{n}\\) curve (blue line), and the gap shows the cost of being too aggressive with small samples.

<iframe src="assets/interactive/se_vs_n_simulation.html" width="100%" height="480" frameborder="0" loading="lazy" title="Interactive SE vs n simulation"></iframe>

*Note: The visualization shows how the sampling distribution becomes narrower (smaller standard error) as sample size increases. This demonstrates the relationship between sample size and estimation precision.*

## Real-world Applications

### 1. Quality Control in Manufacturing

**Single hour's sample vs spec band**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def quality_control_demo():
    """
    Simulate quality control in manufacturing
    """
    # Target specification: 100 ± 2 units
    target = 100
    tolerance = 2

    # Production line measurements (30 samples per hour)
    measurements = np.random.normal(100.5, 1.5, 30)
    mean = np.mean(measurements)
    se = stats.sem(measurements)

    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.hist(measurements, bins=15, alpha=0.7, label='Measurements')
    plt.axvline(mean, color='red', linestyle='--', label='Sample Mean')
    plt.axvline(target, color='green', linestyle=':', label='Target')
    plt.axvline(target + tolerance, color='orange', linestyle=':', label='Tolerance')
    plt.axvline(target - tolerance, color='orange', linestyle=':')
    plt.fill_between([target-tolerance, target+tolerance], [0, 0], [10, 10],
                     color='orange', alpha=0.2)
    plt.title('Quality Control Measurements')
    plt.xlabel('Measurement Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('docs/4-stat-analysis/4.1-inferential-stats/assets/quality_control.png')
    plt.close()

    print("\nQuality Control Report")
    print(f"Specification: {target} ± {tolerance}")
    print(f"Sample Mean: {mean:.2f}")
    print(f"Standard Error: {se:.3f}")
    print(f"Status: {'In Control' if abs(mean - target) <= tolerance else 'Out of Control'}")

quality_control_demo()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="5-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Spec limits</span>
    </div>
    <div class="code-callout__body">
      <p>Define the nominal target (100) and acceptable tolerance band (±2 units) for the production specification.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="9-11" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Hourly sample</span>
    </div>
    <div class="code-callout__body">
      <p>Simulate 30 production measurements drawn from a slightly off-target normal and compute the mean and standard error.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-28" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Control chart</span>
    </div>
    <div class="code-callout__body">
      <p>Plot the histogram with lines for the sample mean, nominal target, and tolerance bounds shaded in orange.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="30-34" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Status report</span>
    </div>
    <div class="code-callout__body">
      <p>Print mean, SE, and a simple "In Control / Out of Control" status based on whether the mean falls within tolerance.</p>
    </div>
  </div>
</aside>
</div>

![Quality control](assets/quality_control.png)
*Figure 4: One hour's sample of 30 measurements (blue bars). Red dashed line = sample mean; green dotted = target (100); orange dotted = tolerance band (±2). The process is "in control" if the sample mean falls inside the orange band.*

*Note: The visualization shows the distribution of quality control measurements with the target value and tolerance limits. This helps us understand if the production process is in control.*

### 2. Political Polling

**Monte Carlo distribution of \\(\hat p\\) and one poll's MOE**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def polling_demo():
    """
    Simulate political polling
    """
    # True population support: 52%
    true_support = 0.52
    sample_size = 1000

    # Simulate multiple polls
    n_polls = 100
    poll_results = [
        np.mean(np.random.binomial(1, true_support, sample_size))
        for _ in range(n_polls)
    ]

    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.hist(poll_results, bins=20, alpha=0.7, label='Poll Results')
    plt.axvline(true_support, color='red', linestyle='--', label='True Support')
    plt.axvline(np.mean(poll_results), color='blue', linestyle=':', label='Mean Support')
    plt.title('Distribution of Poll Results')
    plt.xlabel('Support Proportion')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('docs/4-stat-analysis/4.1-inferential-stats/assets/polling_results.png')
    plt.close()

    # Calculate statistics for a single poll
    poll = np.random.binomial(1, true_support, sample_size)
    p_hat = np.mean(poll)
    se = np.sqrt(p_hat * (1-p_hat) / sample_size)

    print("\nPolitical Poll Results")
    print(f"Support: {p_hat:.1%}")
    print(f"Margin of Error (95% CI): ±{1.96*se:.1%}")
    print(f"Sample Size: {sample_size:,}")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="5-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Poll parameters</span>
    </div>
    <div class="code-callout__body">
      <p>Set the true support level at 52% and simulate 100 independent polls of 1,000 voters each.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="9-14" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Sampling distribution of p̂</span>
    </div>
    <div class="code-callout__body">
      <p>List comprehension builds 100 Bernoulli-draw means, creating an empirical sampling distribution of the proportion.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="16-27" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Poll histogram</span>
    </div>
    <div class="code-callout__body">
      <p>Visualize the spread of poll results with vertical lines marking the true support and the mean of simulated polls.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="29-36" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Single poll margin</span>
    </div>
    <div class="code-callout__body">
      <p>Compute p̂ and the normal-approximation SE for one poll, then print the margin of error (±1.96·SE) as a percentage.</p>
    </div>
  </div>
</aside>
</div>




![Polling results](assets/polling_results.png)
*Figure 5: Histogram of 100 simulated polls of 1,000 voters each. The true support is 52% (red dashed). Most polls land within a couple of percentage points, that wobble is the sampling distribution of \\(\hat p\\).*

*Note: The visualization shows the distribution of poll results from multiple samples. This helps us understand the variability in polling estimates and the role of sampling error.*

## Common misconceptions to clear up

### 1. Sampling Distribution vs. Sample Distribution

- Sample Distribution: The spread of values in ONE sample
- Sampling Distribution: The spread of statistics from MANY samples

### 2. Standard Deviation vs. Standard Error

- Standard Deviation: Spread of individual values
- Standard Error: Spread of sample statistics

### 3. Sample Size Effects

- **Misconception:** "Larger samples always give the right answer." Larger samples reduce *random* error but cannot fix a biased design.
- **Reality:** Larger samples give more *precise* estimates of whatever the design measures, accurate or not.

## Interactive Learning: Try It Yourself

### Mini-Exercise: The Sampling Game

**One draw + approximate 95% band (z = 1.96)**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def sampling_game(true_mean=100, true_std=15, sample_size=30):
    """
    Interactive demonstration of sampling variability
    """
    population = np.random.normal(true_mean, true_std, 10000)
    sample = np.random.choice(population, size=sample_size)
    sample_mean = np.mean(sample)
    se = np.std(sample) / np.sqrt(sample_size)

    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.hist(population, bins=50, alpha=0.3, label='Population')
    plt.hist(sample, bins=15, alpha=0.7, label='Sample')
    plt.axvline(true_mean, color='red', linestyle='--', label='True Mean')
    plt.axvline(sample_mean, color='blue', linestyle=':', label='Sample Mean')
    plt.fill_between([sample_mean-1.96*se, sample_mean+1.96*se], [0, 0], [100, 100],
                     color='blue', alpha=0.2, label='95% CI')
    plt.title('The Sampling Game')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('docs/4-stat-analysis/4.1-inferential-stats/assets/sampling_game.png')
    plt.close()

    print("\nThe Sampling Game")
    print(f"Sample Mean: {sample_mean:.1f}")
    print(f"Standard Error: {se:.2f}")
    print(f"95% CI: ({sample_mean - 1.96*se:.1f}, {sample_mean + 1.96*se:.1f})")
    print(f"Contains true mean? {'Yes' if true_mean-1.96*se <= sample_mean <= true_mean+1.96*se else 'No'}")

sampling_game()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="4-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">One draw</span>
    </div>
    <div class="code-callout__body">
      <p>Generate a population and take one random sample; compute the sample mean and approximate SE using the sample SD.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="10-23" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Overlaid histograms</span>
    </div>
    <div class="code-callout__body">
      <p>Plot the population (transparent) and sample (opaque) with vertical lines for both means and a shaded approximate 95% CI band.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="25-29" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Coverage check</span>
    </div>
    <div class="code-callout__body">
      <p>Print the CI and check whether the true mean falls inside it, a simple illustration of what "95% confidence" means in one run.</p>
    </div>
  </div>
</aside>
</div>

![Sampling game](assets/sampling_game.png)
*Figure 6: One run of the sampling game. Grey histogram = population, blue histogram = the single sample, red dashed = the true mean, blue dotted = our sample mean, blue band = the approximate 95% CI for the mean. Run the code repeatedly and count how often the band covers the red line, it should be close to 95 out of 100 times.*

#### Interactive: 30 samples, 30 intervals

The widget below shows 30 independent samples drawn from the same population, each with its 95% CI. Blue intervals **cover** the true mean (red dashed line); red intervals **miss**. The header reports the coverage rate, across many runs it converges to 95%.

<iframe src="assets/interactive/sampling_game_simulation.html" width="100%" height="540" frameborder="0" loading="lazy" title="Interactive sampling game"></iframe>

This is the *operational* meaning of "95% confidence", not "this specific interval has a 95% probability of being right," but "this *procedure* gets it right 95% of the time."

*Note: The visualization shows how a single sample relates to the population distribution. The confidence interval helps us understand the uncertainty in our sample mean estimate.*

## Practice Questions

Try each question on your own first, then expand the answer to check.

**1.** A sample of 100 customers shows mean spending of $85 with SE = $5. What's the 95% CI?

<details>
<summary>Show answer</summary>

For \\(n = 100\\) the t critical value is essentially 1.98, close enough to the rule-of-thumb \\(z = 1.96\\), so:

\\[
\text{95\% CI} \approx \bar x \pm 1.96 \cdot SE = 85 \pm 1.96 \times 5 = 85 \pm 9.8
\\]

That gives **roughly $75.20 to $94.80**. Plain reading: based on this sample, the procedure used produces an interval that captures the true average customer spend about 95% of the time, and this particular interval is $75.20-$94.80.

</details>

**2.** How would doubling sample size affect the standard error? Show the math.

<details>
<summary>Show answer</summary>

\\(SE = \dfrac{\sigma}{\sqrt{n}}\\). If \\(n\\) becomes \\(2n\\):

\\[
SE_{\text{new}} = \dfrac{\sigma}{\sqrt{2n}} = \dfrac{1}{\sqrt{2}} \cdot \dfrac{\sigma}{\sqrt{n}} \approx 0.707 \cdot SE_{\text{old}}
\\]

So doubling \\(n\\) shrinks the SE by a factor of \\(1/\sqrt{2} \approx 0.71\\), a **~29% reduction**, *not* 50%. To halve the SE you need to **quadruple** the sample size.

</details>

**3.** Why might the CLT not work well with very small samples?

<details>
<summary>Show answer</summary>

The CLT promises that the sampling distribution of the mean is approximately normal **as \\(n\\) grows**. With small \\(n\\):

- A handful of extreme values can dominate the average, so individual sample means swing wildly.
- If the population is heavily skewed (income, waiting times) or has long tails, \\(n = 5\\) or \\(n = 10\\) is far too small for the bell-curve approximation to hold.
- Inferences (CIs, p-values) that rely on the normal approximation will be off, typically too narrow / too confident.

Rule of thumb: aim for \\(n \geq 30\\) for mildly skewed populations, \\(n \geq 50\text{-}100\\) for strongly skewed ones (see the table earlier in the lesson).

</details>

**4.** Design a sampling strategy for estimating average daily website traffic.

<details>
<summary>Show answer</summary>

A reasonable plan:

1. **Define the population clearly**: e.g., "daily unique visitors to the site for a calendar year." Decide whether bots are included.
2. **Pick the sampling unit**: usually the *day*, not individual visits, so each draw is one day's traffic count.
3. **Use stratified sampling by day-of-week**: traffic on Mondays and weekends behaves very differently. Strata: Mon, Tue, Wed, Thu, Fri, Sat, Sun. Sample several days from each.
4. **Cover seasonality**: sample across the whole year (or at least several months) to avoid period-specific bias (e.g., holiday spikes).
5. **Set sample size from desired precision**: choose how tight you want the CI for the average and back-solve via \\(n = (z\sigma/\text{MOE})^2\\) using a pilot estimate of the day-to-day standard deviation.
6. **Report \\(\bar x\\) with a confidence interval**, not just a point estimate, so consumers see the uncertainty.

If you have analytics for *every* day, you don't need to sample at all, just compute the mean directly. Sampling matters when measurement is costly (e.g., manual log review, paid tools with quotas).

</details>

**5.** How would you explain sampling distributions to a non-technical stakeholder?

<details>
<summary>Show answer</summary>

> "Imagine we ran our survey 100 different times, each with a fresh group of customers. We'd get 100 slightly different averages, that's not a mistake, that's just life. The *sampling distribution* is the picture of those 100 averages.
>
> Two things to know:
>
> 1. The averages cluster around the true value, not on top of it. So our single survey number is *near* the truth, not exactly the truth.
> 2. The bigger our survey, the tighter that cluster, meaning our number is closer to the truth.
>
> That's why we report a *range* (a confidence interval) instead of a single number: it's an honest way of saying 'here's our best guess, and here's how much that guess could move if we redid the survey'."

</details>

## Key Takeaways

1. Sampling distributions help us understand estimation uncertainty
2. The Central Limit Theorem is a powerful tool for inference
3. Standard error decreases with larger sample sizes
4. Different statistics have different sampling distributions
5. Visualizing sampling distributions aids understanding
6. Real-world applications include quality control and polling
7. Common misconceptions can lead to incorrect interpretations

## Gotchas

- **Confusing the sampling distribution with the sample distribution**: the *sample distribution* is the histogram of values in one dataset; the *sampling distribution* is the distribution of a statistic (e.g., x̄) across hypothetical repeated samples. The CLT applies to the second, not the first.
- **Believing the CLT means "your data become normal with more observations"**: the CLT says the *sampling distribution of the mean* becomes approximately normal; the raw data can stay skewed or bimodal regardless of n. Using CLT to justify normality of individual values leads to wrong model choices downstream.
- **Using `np.random.choice(population, size=n)` to build a sampling distribution when n is small**: for a population of size 10,000 and n=5, each resample is highly sensitive to outliers and the normal approximation is poor. The lesson's threshold of n=30 is a common rule of thumb; skewed populations may need n>50 before the CLT kicks in.
- **Equating standard deviation with standard error**: `np.std(data)` gives the spread of individual observations; `scipy.stats.sem(data)` gives how much the *mean* varies across samples. Using the wrong one in a CI formula inflates or deflates the interval by a factor of √n.
- **Overlaying a normal curve on a sampling distribution and concluding CLT "proved"**: the lesson's `stats.norm.pdf` overlay is fitted to the simulated means, so it will always look like a decent fit. A formal normality check (Shapiro-Wilk or Q-Q plot) is needed to verify CLT has kicked in adequately for a given n and population shape.
- **Running the demonstration code without a fixed seed and expecting stable output**: the lesson's CLT simulation uses `np.random.choice` inside a loop without a per-run seed; re-running will produce slightly different empirical SEs. Always seed before benchmarking or sharing, and report that values are approximate.

## Next steps

- Continue to [Confidence intervals](./confidence-intervals.md), where the σ/√n formula you just met becomes the margin of error.

## Additional Resources

- [Interactive Sampling Distribution Simulator](https://seeing-theory.brown.edu/frequentist-inference/index.html)
- [Understanding Sampling Distributions](https://statisticsbyjim.com/basics/sampling-distribution/)
- [CLT in Practice](https://www.khanacademy.org/math/statistics-probability/sampling-distributions-library)

Remember: Sampling distributions are the foundation of statistical inference. Understanding them helps us make better decisions with data!
