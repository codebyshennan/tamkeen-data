# Probability Distributions with Python

**After this lesson:** you can explain Probability Distributions with Python and try the examples in your own notebook.

### Video

<div class="video-embed">
<iframe width="560" height="315" src="https://www.youtube.com/embed/iYiOVISeS84" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

*StatQuest with Josh Starmer, The normal distribution, clearly explained*

## Understanding random variables through code

A **random variable** is a quantity whose value is uncertain until you observe it (or simulate it). You describe it with a **distribution**: either a list of outcomes with probabilities (**discrete**) or a density over a continuum (**continuous**). The same code pattern appears everywhere: **specify law**, **draw samples**, **plot** to see shape.

### Implementing random variables

we will look at random variables using Python:

**`RandomVariableExplorer`: discrete vs continuous draws**

- **Purpose:** Tie code to the idea of a random variable: **simulate** draws from a discrete law (`np.random.choice` with probabilities) and from continuous families (`normal`, `uniform`), then **plot** with bar vs histogram/KDE.
- **Walkthrough:** `simulate_discrete` uses `p=`; `simulate_continuous` branches on `distribution`; `plot_distribution` picks `discrete` vs `continuous` from `n_unique`.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict, Tuple, Optional

class RandomVariableExplorer:
    """Explore and visualize random variables"""

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize explorer with optional seed"""
        if random_seed is not None:
            np.random.seed(random_seed)

    def simulate_discrete(
        self,
        values: List[int],
        probabilities: List[float],
        n_samples: int = 1000
    ) -> pd.Series:
        """Simulate discrete random variable"""
        samples = np.random.choice(
            values,
            size=n_samples,
            p=probabilities
        )
        return pd.Series(samples, name='Value')

    def simulate_continuous(
        self,
        distribution: str,
        params: Dict[str, float],
        n_samples: int = 1000
    ) -> pd.Series:
        """Simulate continuous random variable"""
        if distribution == 'normal':
            samples = np.random.normal(
                loc=params.get('mean', 0),
                scale=params.get('std', 1),
                size=n_samples
            )
        elif distribution == 'uniform':
            samples = np.random.uniform(
                low=params.get('low', 0),
                high=params.get('high', 1),
                size=n_samples
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        return pd.Series(samples, name='Value')

    def plot_distribution(
        self,
        data: pd.Series,
        kind: str = 'auto'
    ) -> None:
        """Plot distribution of random variable"""
        plt.figure(figsize=(12, 6))

        if kind == 'auto':
            n_unique = len(data.unique())
            kind = 'discrete' if n_unique <= 10 else 'continuous'

        if kind == 'discrete':
            value_counts = data.value_counts(normalize=True)
            plt.bar(value_counts.index, value_counts.values, alpha=0.8)
            plt.xlabel('Value')
            plt.ylabel('Probability')
        else:
            sns.histplot(data, stat='density', kde=True, alpha=0.5)
            plt.xlabel('Value')
            plt.ylabel('Density')

        plt.title('Distribution of Random Variable')
        plt.grid(True, alpha=0.3)
        plt.show()
        print("\nSummary Statistics:")
        print(data.describe().round(3))

# Example usage
explorer = RandomVariableExplorer(random_seed=42)
die_rolls = explorer.simulate_discrete([1,2,3,4,5,6], [1/6]*6, n_samples=1000)
print("\nDie Rolls:")
explorer.plot_distribution(die_rolls, kind='discrete')

heights = explorer.simulate_continuous('normal', {'mean': 170, 'std': 10}, n_samples=1000)
print("\nHeight Distribution:")
explorer.plot_distribution(heights, kind='continuous')
{% endhighlight %}

<figure>
<img src="assets/probability-distributions_fig_1.png" alt="probability-distributions" />
<figcaption>Figure 1: Distribution of Random Variable</figcaption>
</figure>


<figure>
<img src="assets/probability-distributions_fig_2.png" alt="probability-distributions" />
<figcaption>Figure 2: Distribution of Random Variable</figcaption>
</figure>

```

Die Rolls:

Summary Statistics:
count    1000.000
mean        3.443
std         1.725
min         1.000
25%         2.000
50%         3.000
75%         5.000
max         6.000
Name: Value, dtype: float64

Height Distribution:

Summary Statistics:
count    1000.000
mean      170.989
std         9.889
min       140.786
25%       164.359
50%       170.842
75%       177.396
max       201.931
Name: Value, dtype: float64
```


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports</span>
    </div>
    <div class="code-callout__body">
      <p>Imports NumPy, pandas, Matplotlib, Seaborn, SciPy stats, and typing, the full stack needed for simulation, analysis, and plotting.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="9-28" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Discrete Simulation</span>
    </div>
    <div class="code-callout__body">
      <p>Seeds the RNG if requested, then draws samples from a discrete law using <code>np.random.choice</code> with explicit probabilities.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="30-53" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Continuous Simulation</span>
    </div>
    <div class="code-callout__body">
      <p>Branches on the distribution name to call the matching NumPy generator, normal or uniform, and raises if the name is unknown.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="55-88" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Plot and Demo</span>
    </div>
    <div class="code-callout__body">
      <p>Auto-detects discrete vs continuous from unique-value count, then plots bar chart or histogram+KDE; the bottom lines demo both die rolls and heights.</p>
    </div>
  </div>
</aside>
</div>



```

Die Rolls:

Summary Statistics:
count    1000.000
mean        3.443
std         1.725
min         1.000
25%         2.000
50%         3.000
75%         5.000
max         6.000
Name: Value, dtype: float64

Height Distribution:

Summary Statistics:
count    1000.000
mean      170.989
std         9.889
min       140.786
25%       164.359
50%       170.842
75%       177.396
max       201.931
Name: Value, dtype: float64
```

---

### Expected Value and Variance

Implement tools for calculating distribution properties:

**Moments and skew/kurtosis on samples**

- **Purpose:** Connect **E[X]** and **Var(X)** for both tabulated `(values, probabilities)` and raw samples; visualize with histogram + mean/median and a normal Q-Q plot.
- **Walkthrough:** `calculate_expected_value` / `calculate_variance` use `np.mean`/`np.var` when `probabilities` is `None`; `analyze_distribution` builds the summary dict and `stats.probplot` for Q-Q.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class DistributionAnalyzer:
    """Analyze properties of distributions"""

    @staticmethod
    def calculate_expected_value(
        values: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ) -> float:
        """Calculate expected value"""
        if probabilities is None:
            return np.mean(values)
        return np.sum(values * probabilities)

    @staticmethod
    def calculate_variance(
        values: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        ddof: int = 0
    ) -> float:
        """Calculate variance"""
        if probabilities is None:
            return np.var(values, ddof=ddof)
        expected_value = DistributionAnalyzer.calculate_expected_value(
            values, probabilities
        )
        squared_deviations = (values - expected_value) ** 2
        return np.sum(squared_deviations * probabilities)

    @staticmethod
    def calculate_skewness(data: np.ndarray) -> float:
        return stats.skew(data)

    @staticmethod
    def calculate_kurtosis(data: np.ndarray) -> float:
        return stats.kurtosis(data)

    def analyze_distribution(self, data: np.ndarray, name: str = "Distribution") -> None:
        """Print comprehensive distribution analysis"""
        analysis = {
            'Mean': np.mean(data), 'Median': np.median(data),
            'Std Dev': np.std(data), 'Variance': np.var(data),
            'Skewness': self.calculate_skewness(data),
            'Kurtosis': self.calculate_kurtosis(data)
        }
        print(f"\n{name} Analysis:")
        for metric, value in analysis.items():
            print(f"{metric}: {value:.3f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        sns.histplot(data, kde=True, ax=ax1)
        ax1.axvline(analysis['Mean'], color='r', linestyle='--', label='Mean')
        ax1.axvline(analysis['Median'], color='g', linestyle='--', label='Median')
        ax1.set_title('Distribution Plot')
        ax1.legend()
        stats.probplot(data, plot=ax2)
        ax2.set_title('Q-Q Plot')
        plt.tight_layout()
        plt.show()

# Example usage
analyzer = DistributionAnalyzer()
print("\nNormal Distribution:")
normal_data = np.random.normal(loc=0, scale=1, size=1000)
analyzer.analyze_distribution(normal_data, "Normal")

print("\nRight-Skewed Distribution:")
right_skewed = np.random.lognormal(mean=0, sigma=1, size=1000)
analyzer.analyze_distribution(right_skewed, "Right-Skewed")

print("\nUniform Distribution:")
uniform_data = np.random.uniform(low=-3, high=3, size=1000)
analyzer.analyze_distribution(uniform_data, "Uniform")
{% endhighlight %}

<figure>
<img src="assets/probability-distributions_fig_3.png" alt="probability-distributions" />
<figcaption>Figure 3: Distribution Plot</figcaption>
</figure>


<figure>
<img src="assets/probability-distributions_fig_4.png" alt="probability-distributions" />
<figcaption>Figure 4: Distribution Plot</figcaption>
</figure>


<figure>
<img src="assets/probability-distributions_fig_5.png" alt="probability-distributions" />
<figcaption>Figure 5: Distribution Plot</figcaption>
</figure>

```

Normal Distribution:

Normal Analysis:
Mean: 0.014
Median: 0.011
Std Dev: 0.970
Variance: 0.941
Skewness: 0.002
Kurtosis: 0.052

Right-Skewed Distribution:

Right-Skewed Analysis:
Mean: 1.702
Median: 0.969
Std Dev: 2.564
Variance: 6.572
Skewness: 8.941
Kurtosis: 142.635

Uniform Distribution:

Uniform Analysis:
Mean: -0.024
Median: -0.098
Std Dev: 1.736
Variance: 3.014
Skewness: 0.054
Kurtosis: -1.214
```


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-12" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Expected Value</span>
    </div>
    <div class="code-callout__body">
      <p>Returns <code>np.mean</code> for raw samples when no probabilities are given, or the weighted sum <code>Σ(x·p)</code> for a tabulated discrete distribution.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="14-27" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Variance Calculation</span>
    </div>
    <div class="code-callout__body">
      <p>Mirrors the expected-value duality: <code>np.var</code> for samples, or <code>Σ((x−μ)²·p)</code> using the previously computed mean for discrete distributions.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="29-57" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Full Analysis and Plot</span>
    </div>
    <div class="code-callout__body">
      <p>Builds a summary dict of six moments, prints them, then plots a histogram+KDE with mean/median lines alongside a Q-Q plot for normality assessment.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="59-68" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Demo Usage</span>
    </div>
    <div class="code-callout__body">
      <p>Runs the analyzer on three distribution shapes, symmetric normal, right-skewed lognormal, and uniform, so you can compare their outputs side by side.</p>
    </div>
  </div>
</aside>
</div>




```

Normal Distribution:

Normal Analysis:
Mean: 0.014
Median: 0.011
Std Dev: 0.970
Variance: 0.941
Skewness: 0.002
Kurtosis: 0.052

Right-Skewed Distribution:

Right-Skewed Analysis:
Mean: 1.702
Median: 0.969
Std Dev: 2.564
Variance: 6.572
Skewness: 8.941
Kurtosis: 142.635

Uniform Distribution:

Uniform Analysis:
Mean: -0.024
Median: -0.098
Std Dev: 1.736
Variance: 3.014
Skewness: 0.054
Kurtosis: -1.214
```

## Common Probability Distributions

---

### Implementing Distribution Functions

Create tools for working with common distributions:

**Sampling binomial, Poisson, normal, exponential**

- **Purpose:** See how NumPy's `np.random.*` generators map to common families; compare shapes side-by-side with histograms and normal Q-Q panels.
- **Walkthrough:** Each method wraps one generator (`binomial`, `poisson`, `normal`, `exponential`); `plot_distributions` lays out two columns per distribution.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class ProbabilityDistributions:
    """Work with common probability distributions"""

    def __init__(self, random_seed: Optional[int] = None):
        if random_seed is not None:
            np.random.seed(random_seed)

    def binomial(self, n: int, p: float, size: int = 1000) -> pd.Series:
        """Generate binomial distribution"""
        return pd.Series(np.random.binomial(n, p, size), name='Binomial')

    def poisson(self, lambda_: float, size: int = 1000) -> pd.Series:
        """Generate Poisson distribution"""
        return pd.Series(np.random.poisson(lambda_, size), name='Poisson')

    def normal(self, mean: float, std: float, size: int = 1000) -> pd.Series:
        """Generate normal distribution"""
        return pd.Series(np.random.normal(mean, std, size), name='Normal')

    def exponential(self, scale: float, size: int = 1000) -> pd.Series:
        """Generate exponential distribution"""
        return pd.Series(np.random.exponential(scale, size), name='Exponential')

    def plot_distributions(self, distributions: Dict[str, pd.Series]) -> None:
        """Plot multiple distributions"""
        n_dist = len(distributions)
        fig, axes = plt.subplots(n_dist, 2, figsize=(15, 5 * n_dist))

        for i, (name, data) in enumerate(distributions.items()):
            sns.histplot(data, kde=True, ax=axes[i, 0])
            axes[i, 0].set_title(f'{name} Distribution')
            stats.probplot(data, dist='norm', plot=axes[i, 1])
            axes[i, 1].set_title(f'{name} Q-Q Plot')

        plt.tight_layout()
        plt.show()

        print("\nSummary Statistics:")
        for name, data in distributions.items():
            print(f"\n{name}:")
            print(data.describe().round(3))

# Example usage
pd_explorer = ProbabilityDistributions(random_seed=42)
distributions = {
    'Binomial': pd_explorer.binomial(n=10, p=0.5),
    'Poisson': pd_explorer.poisson(lambda_=3),
    'Normal': pd_explorer.normal(mean=0, std=1),
    'Exponential': pd_explorer.exponential(scale=2)
}
pd_explorer.plot_distributions(distributions)
{% endhighlight %}

<figure>
<img src="assets/probability-distributions_fig_6.png" alt="probability-distributions" />
<figcaption>Figure 6: Binomial Distribution</figcaption>
</figure>

```

Summary Statistics:

Binomial:
count    1000.000
mean        4.939
std         1.579
min         1.000
25%         4.000
50%         5.000
75%         6.000
max        10.000
Name: Binomial, dtype: float64

Poisson:
count    1000.000
mean        2.979
std         1.693
min         0.000
25%         2.000
50%         3.000
75%         4.000
max         9.000
Name: Poisson, dtype: float64

Normal:
count    1000.000
mean       -0.014
std         0.980
min        -3.275
25%        -0.671
50%        -0.060
75%         0.618
max         2.769
Name: Normal, dtype: float64

Exponential:
count    1000.000
mean        1.980
std         1.917
min         0.000
25%         0.577
50%         1.362
75%         2.865
max        14.280
Name: Exponential, dtype: float64
```


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-21" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Four Generators</span>
    </div>
    <div class="code-callout__body">
      <p>Each method wraps one NumPy generator, binomial, Poisson, normal, exponential, and returns a named pandas Series for easy labelling in plots.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="23-40" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Grid Plot</span>
    </div>
    <div class="code-callout__body">
      <p>Creates a 2-column grid with one row per distribution: histogram+KDE on the left and a Q-Q plot on the right for normality comparison.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="42-49" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Demo Run</span>
    </div>
    <div class="code-callout__body">
      <p>Generates one sample from each family with a fixed seed, then passes the dict to <code>plot_distributions</code> so all four appear in the same figure.</p>
    </div>
  </div>
</aside>
</div>


```

Summary Statistics:

Binomial:
count    1000.000
mean        4.939
std         1.579
min         1.000
25%         4.000
50%         5.000
75%         6.000
max        10.000
Name: Binomial, dtype: float64

Poisson:
count    1000.000
mean        2.979
std         1.693
min         0.000
25%         2.000
50%         3.000
75%         4.000
max         9.000
Name: Poisson, dtype: float64

Normal:
count    1000.000
mean       -0.014
std         0.980
min        -3.275
25%        -0.671
50%        -0.060
75%         0.618
max         2.769
Name: Normal, dtype: float64

Exponential:
count    1000.000
mean        1.980
std         1.917
min         0.000
25%         0.577
50%         1.362
75%         2.865
max        14.280
Name: Exponential, dtype: float64
```

---

### Distribution Shape Analysis

Create tools for analyzing distribution shapes:

**Classify skew/tails and compare plot types**

- **Purpose:** Practice reading **skewness** and **kurtosis** thresholds, and pair histograms with box and violin plots for the same data.
- **Walkthrough:** `classify_shape` uses `stats.skew` / `stats.kurtosis`; `plot_shape_analysis` builds a 2×2 grid with `sns.histplot`, `sns.boxplot`, `sns.violinplot`.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class ShapeAnalyzer:
    """Analyze and classify distribution shapes"""

    @staticmethod
    def classify_shape(data: np.ndarray) -> str:
        """Classify distribution shape by skewness and kurtosis"""
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)

        if abs(skewness) < 0.5:
            shape = "Approximately Symmetric"
        elif skewness > 0:
            shape = "Right-Skewed"
        else:
            shape = "Left-Skewed"

        if kurtosis > 1:
            shape += ", Heavy-Tailed"
        elif kurtosis < -1:
            shape += ", Light-Tailed"

        return shape

    def plot_shape_analysis(
        self, data: np.ndarray,
        title: str = "Distribution Shape Analysis"
    ) -> None:
        """Create comprehensive shape analysis plot"""
        shape = self.classify_shape(data)
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)

        ax1 = fig.add_subplot(gs[0, :])
        sns.histplot(data, kde=True, ax=ax1)
        ax1.set_title(f'{title}\nClassified as: {shape}')
        mean, median = np.mean(data), np.median(data)
        ax1.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
        ax1.axvline(median, color='g', linestyle='--', label=f'Median: {median:.2f}')
        ax1.legend()

        ax2 = fig.add_subplot(gs[1, 0])
        sns.boxplot(y=data, ax=ax2)
        ax2.set_title('Box Plot')

        ax3 = fig.add_subplot(gs[1, 1])
        sns.violinplot(y=data, ax=ax3)
        ax3.set_title('Violin Plot')

        plt.tight_layout()
        plt.show()
        print("\nShape Statistics:")
        print(f"Skewness: {stats.skew(data):.3f}")
        print(f"Kurtosis: {stats.kurtosis(data):.3f}")

# Example usage
shape_analyzer = ShapeAnalyzer()
print("\nNormal Distribution:")
shape_analyzer.plot_shape_analysis(np.random.normal(0, 1, 1000), "Normal Distribution")

print("\nRight-Skewed Distribution:")
shape_analyzer.plot_shape_analysis(np.random.lognormal(0, 1, 1000), "Right-Skewed Distribution")

print("\nBimodal Distribution:")
bimodal = np.concatenate([np.random.normal(-2, 0.5, 500), np.random.normal(2, 0.5, 500)])
shape_analyzer.plot_shape_analysis(bimodal, "Bimodal Distribution")
{% endhighlight %}

<figure>
<img src="assets/probability-distributions_fig_7.png" alt="probability-distributions" />
<figcaption>Figure 7: Normal Distribution
Classified as: Approximately Symmetric</figcaption>
</figure>


<figure>
<img src="assets/probability-distributions_fig_8.png" alt="probability-distributions" />
<figcaption>Figure 8: Right-Skewed Distribution
Classified as: Right-Skewed, Heavy-Tailed</figcaption>
</figure>


<figure>
<img src="assets/probability-distributions_fig_9.png" alt="probability-distributions" />
<figcaption>Figure 9: Bimodal Distribution
Classified as: Approximately Symmetric, Light-Tailed</figcaption>
</figure>

```

Normal Distribution:

Shape Statistics:
Skewness: 0.054
Kurtosis: -0.093

Right-Skewed Distribution:

Shape Statistics:
Skewness: 4.106
Kurtosis: 28.778

Bimodal Distribution:

Shape Statistics:
Skewness: 0.005
Kurtosis: -1.763
```


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-21" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Shape Classifier</span>
    </div>
    <div class="code-callout__body">
      <p>Uses SciPy's skew and kurtosis to classify the distribution, left/right/symmetric for skewness, and heavy/light-tailed for kurtosis, and appends both labels.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="23-49" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Composite Plot</span>
    </div>
    <div class="code-callout__body">
      <p>Uses <code>GridSpec</code> to place a full-width histogram+KDE on top with mean/median lines, then a box plot and violin plot side-by-side below.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="51-60" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Demo: Three Shapes</span>
    </div>
    <div class="code-callout__body">
      <p>Runs the analyzer on a symmetric normal, a right-skewed lognormal, and a hand-crafted bimodal so you can compare how each shape reads in the plots.</p>
    </div>
  </div>
</aside>
</div>




```

Normal Distribution:

Shape Statistics:
Skewness: 0.054
Kurtosis: -0.093

Right-Skewed Distribution:

Shape Statistics:
Skewness: 4.106
Kurtosis: 28.778

Bimodal Distribution:

Shape Statistics:
Skewness: 0.005
Kurtosis: -1.763
```

## Practice Exercises

Try these distribution analysis exercises:

1. **Stock Returns Analysis**

   - **Purpose:** Stub for **Practice Exercise 1**-implement the four comment bullets (load prices, returns, fit, tails) using your own data source.

   ```python
   # Create functions to:
   # - Load stock price data
   # - Calculate daily returns
   # - Fit distribution to returns
   # - Analyze tail behavior
   ```

2. **Customer Behavior Model**

   - **Purpose:** Stub for **Practice Exercise 2**-model frequency and order value distributions and lifetime-style summaries from transactional data.

   ```python
   # Build analysis tools for:
   # - Purchase frequency distribution
   # - Order value distribution
   # - Customer lifetime modeling
   ```

3. **Quality Control System**

   - **Purpose:** Stub for **Practice Exercise 3**-monitor measurements, compare to baseline distributions, and set control limits.

   ```python
   # Implement system to:
   # - Monitor process measurements
   # - Detect distribution shifts
   # - Calculate control limits
   # - Generate alerts
   ```

Remember:

- Use appropriate distributions
- Validate distribution assumptions
- Consider sample size effects
- Create clear visualizations
- Document your analysis

## Common pitfalls

- **Wrong support**: Binomial counts cannot be negative; Normal models are continuous, check that your data fits the story.
- **Confusing PDF and probability**: For continuous variables, probability comes from areas under the curve, not the height at a point.
- **Small-sample behavior**: Histograms and fitted curves look smoother as **n** grows; don't overfit a distribution from a tiny sample.

## Next steps

Continue to [Probability distribution families](./probability-distribution-families.md), then [Two-variable statistics](./two-variable-statistics.md).

Happy analyzing!
