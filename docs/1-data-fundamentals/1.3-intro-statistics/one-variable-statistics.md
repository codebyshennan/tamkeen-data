# One-Variable Statistics with Python 📊

## Understanding One-Variable Statistics

{% stepper %}
{% step %}
### What is One-Variable Statistics?
One-variable (univariate) statistics helps us understand individual variables in our dataset. Let's explore with Python:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset: Student test scores
scores = np.array([75, 82, 95, 68, 90, 88, 76, 89, 94, 83])

# Create a pandas Series for better analysis
scores_series = pd.Series(scores, name='Test Scores')

# Basic summary
print("Summary Statistics:")
print(scores_series.describe())

# Visualize distribution
plt.figure(figsize=(10, 6))
sns.histplot(scores_series, kde=True)
plt.title('Distribution of Test Scores')
plt.show()
```

This gives us a quick overview of:
- Central tendency (mean, median)
- Spread (std, quartiles)
- Distribution shape
{% endstep %}

{% step %}
### Real-World Applications
Let's analyze real estate data:

```python
# Sample house prices (in thousands)
house_prices = pd.Series([
    250, 280, 295, 310, 460, 475, 
    285, 290, 310, 330, 380, 400
], name='House Prices ($K)')

def analyze_distribution(data: pd.Series) -> None:
    """Analyze and visualize data distribution"""
    # Calculate statistics
    stats = {
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'skewness': data.skew(),
        'kurtosis': data.kurtosis()
    }
    
    # Print statistics
    print("\nDistribution Analysis:")
    for stat, value in stats.items():
        print(f"{stat.title()}: {value:.2f}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram with KDE
    sns.histplot(data, kde=True, ax=ax1)
    ax1.axvline(stats['mean'], color='r', linestyle='--', 
                label=f"Mean: {stats['mean']:.2f}")
    ax1.axvline(stats['median'], color='g', linestyle='--', 
                label=f"Median: {stats['median']:.2f}")
    ax1.legend()
    ax1.set_title('Distribution with Mean and Median')
    
    # Boxplot
    sns.boxplot(y=data, ax=ax2)
    ax2.set_title('Boxplot with Quartiles')
    
    plt.tight_layout()
    plt.show()

# Analyze house prices
analyze_distribution(house_prices)
```
{% endstep %}
{% endstepper %}

## Measures of Central Tendency

{% stepper %}
{% step %}
### Mean, Median, and Mode in Python
Let's implement all three measures:

```python
class CentralTendency:
    """Calculate and compare central tendency measures"""
    
    def __init__(self, data: pd.Series):
        self.data = data
        self.results = self._calculate_measures()
    
    def _calculate_measures(self) -> dict:
        """Calculate all central tendency measures"""
        return {
            'mean': self.data.mean(),
            'median': self.data.median(),
            'mode': self.data.mode()[0],
            'trimmed_mean': stats.trim_mean(self.data, 0.1)
        }
    
    def compare_measures(self) -> None:
        """Compare different measures visually"""
        plt.figure(figsize=(10, 6))
        
        # Plot distribution
        sns.histplot(self.data, kde=True)
        
        # Add vertical lines for measures
        colors = ['r', 'g', 'b', 'y']
        labels = ['Mean', 'Median', 'Mode', 'Trimmed Mean']
        
        for (measure, value), color, label in zip(
            self.results.items(), colors, labels
        ):
            plt.axvline(
                value,
                color=color,
                linestyle='--',
                label=f"{label}: {value:.2f}"
            )
        
        plt.title('Distribution with Central Tendency Measures')
        plt.legend()
        plt.show()
    
    def print_summary(self) -> None:
        """Print summary of central tendency measures"""
        print("\nCentral Tendency Measures:")
        for measure, value in self.results.items():
            print(f"{measure.title()}: {value:.2f}")

# Example with salary data
salaries = pd.Series([
    45000, 48000, 51000, 52000, 54000,
    55000, 57000, 58000, 60000, 150000
], name='Salaries')

# Analyze central tendency
ct = CentralTendency(salaries)
ct.print_summary()
ct.compare_measures()
```

💡 **Pro Tip**: Use `trimmed_mean` when your data has outliers but you still want to use a mean-like measure!
{% endstep %}

{% step %}
### When to Use Each Measure
Let's create a function to help choose the appropriate measure:

```python
def recommend_central_measure(data: pd.Series) -> str:
    """Recommend appropriate central tendency measure"""
    # Calculate key statistics
    skewness = data.skew()
    has_outliers = (
        np.abs(stats.zscore(data)) > 3
    ).any()
    is_symmetric = abs(skewness) < 0.5
    
    # Create recommendation
    if is_symmetric and not has_outliers:
        return (
            "Recommend: Mean\n"
            "Reason: Data is symmetric without outliers"
        )
    elif has_outliers:
        return (
            "Recommend: Median\n"
            "Reason: Data contains outliers"
        )
    else:
        return (
            "Recommend: Both Mean and Median\n"
            "Reason: Data is moderately skewed"
        )

# Example usage
datasets = {
    'Symmetric': pd.Series(np.random.normal(100, 10, 1000)),
    'With Outliers': pd.Series([*np.random.normal(100, 10, 99), 500]),
    'Skewed': pd.Series(np.random.exponential(5, 1000))
}

for name, data in datasets.items():
    print(f"\n{name} Dataset:")
    print(recommend_central_measure(data))
```
{% endstep %}
{% endstepper %}

## Measures of Variability

{% stepper %}
{% step %}
### Calculating Spread Measures
Let's create a comprehensive spread analyzer:

```python
class SpreadAnalyzer:
    """Analyze data spread using various measures"""
    
    def __init__(self, data: pd.Series):
        self.data = data
        self.stats = self._calculate_stats()
    
    def _calculate_stats(self) -> dict:
        """Calculate various spread statistics"""
        q1, q3 = self.data.quantile([0.25, 0.75])
        iqr = q3 - q1
        
        return {
            'range': self.data.max() - self.data.min(),
            'std': self.data.std(),
            'variance': self.data.var(),
            'mad': self.data.mad(),  # Mean absolute deviation
            'iqr': iqr,
            'quartiles': {
                'Q1': q1,
                'Q3': q3
            }
        }
    
    def identify_outliers(self) -> pd.Series:
        """Identify outliers using IQR method"""
        q1 = self.stats['quartiles']['Q1']
        q3 = self.stats['quartiles']['Q3']
        iqr = self.stats['iqr']
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        return self.data[
            (self.data < lower_bound) |
            (self.data > upper_bound)
        ]
    
    def plot_spread(self) -> None:
        """Visualize data spread"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Box plot
        sns.boxplot(x=self.data, ax=ax1)
        ax1.set_title('Boxplot with Quartiles')
        
        # Distribution plot with spread measures
        sns.histplot(self.data, kde=True, ax=ax2)
        
        # Add vertical lines for spread measures
        mean = self.data.mean()
        std = self.stats['std']
        
        ax2.axvline(mean, color='r', linestyle='--',
                    label='Mean')
        ax2.axvline(mean + std, color='g', linestyle=':',
                    label='+1 Std Dev')
        ax2.axvline(mean - std, color='g', linestyle=':',
                    label='-1 Std Dev')
        
        ax2.legend()
        ax2.set_title('Distribution with Standard Deviation')
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self) -> None:
        """Print spread analysis summary"""
        print("\nSpread Analysis:")
        for measure, value in self.stats.items():
            if measure != 'quartiles':
                print(f"{measure.title()}: {value:.2f}")
        
        print("\nQuartiles:")
        for q, v in self.stats['quartiles'].items():
            print(f"{q}: {v:.2f}")
        
        outliers = self.identify_outliers()
        if len(outliers) > 0:
            print(f"\nFound {len(outliers)} outliers:")
            print(outliers.values)

# Example with temperature data
temperatures = pd.Series([
    18.5, 19.2, 20.1, 19.8, 20.2, 20.5,
    19.9, 20.3, 25.0, 18.9, 19.5, 20.0
], name='Daily Temperatures')

# Analyze spread
spread = SpreadAnalyzer(temperatures)
spread.print_summary()
spread.plot_spread()
```
{% endstep %}

{% step %}
### Understanding Variability in Context
Let's analyze variability in different scenarios:

```python
def compare_variability(datasets: dict) -> None:
    """Compare variability across different datasets"""
    fig, axes = plt.subplots(
        len(datasets), 2,
        figsize=(15, 5 * len(datasets))
    )
    
    results = {}
    
    for (name, data), (ax1, ax2) in zip(
        datasets.items(), axes
    ):
        # Calculate statistics
        stats = {
            'std': data.std(),
            'iqr': data.quantile(0.75) - data.quantile(0.25),
            'cv': data.std() / data.mean()  # Coefficient of variation
        }
        results[name] = stats
        
        # Create visualizations
        sns.boxplot(x=data, ax=ax1)
        ax1.set_title(f'{name} - Boxplot')
        
        sns.histplot(data, kde=True, ax=ax2)
        ax2.set_title(f'{name} - Distribution')
        
        # Add mean and std dev lines
        mean = data.mean()
        std = stats['std']
        ax2.axvline(mean, color='r', linestyle='--',
                    label='Mean')
        ax2.axvline(mean + std, color='g', linestyle=':',
                    label='+1 Std Dev')
        ax2.axvline(mean - std, color='g', linestyle=':',
                    label='-1 Std Dev')
        ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison
    print("\nVariability Comparison:")
    measures = ['std', 'iqr', 'cv']
    
    for measure in measures:
        print(f"\n{measure.upper()}:")
        for name, stats in results.items():
            print(f"{name}: {stats[measure]:.3f}")

# Example datasets
datasets = {
    'Stock Prices': pd.Series([
        100, 102, 101, 103, 98, 99, 102,
        101, 100, 103, 97, 102
    ]),
    'Temperature': pd.Series([
        20.1, 20.3, 20.2, 20.1, 20.4, 20.2,
        20.3, 20.1, 20.2, 20.3
    ]),
    'Website Traffic': pd.Series([
        1000, 1500, 800, 2000, 1200, 1800,
        900, 2500, 1100, 1300
    ])
}

compare_variability(datasets)
```
{% endstep %}
{% endstepper %}

## Frequency Distributions and Visualization

{% stepper %}
{% step %}
### Creating Frequency Distributions
Let's create a comprehensive frequency analyzer:

```python
class FrequencyAnalyzer:
    """Analyze and visualize frequency distributions"""
    
    def __init__(
        self,
        data: pd.Series,
        bins: Optional[int] = None
    ):
        self.data = data
        self.bins = bins or self._suggest_bins()
        self.freq_dist = self._calculate_frequency()
    
    def _suggest_bins(self) -> int:
        """Suggest number of bins using Sturge's rule"""
        return int(1 + 3.322 * np.log10(len(self.data)))
    
    def _calculate_frequency(self) -> pd.DataFrame:
        """Calculate frequency distribution"""
        # Create bins
        hist, bin_edges = np.histogram(
            self.data, bins=self.bins
        )
        
        # Create frequency table
        freq_df = pd.DataFrame({
            'bin_start': bin_edges[:-1],
            'bin_end': bin_edges[1:],
            'frequency': hist
        })
        
        # Add relative and cumulative frequencies
        freq_df['relative_freq'] = (
            freq_df['frequency'] / len(self.data)
        )
        freq_df['cumulative_freq'] = (
            freq_df['frequency'].cumsum()
        )
        freq_df['cumulative_relative'] = (
            freq_df['relative_freq'].cumsum()
        )
        
        return freq_df
    
    def plot_distributions(self) -> None:
        """Create visualization suite"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=(15, 12)
        )
        
        # Histogram
        sns.histplot(self.data, bins=self.bins, ax=ax1)
        ax1.set_title('Frequency Distribution')
        
        # Relative frequency
        ax2.bar(
            range(len(self.freq_dist)),
            self.freq_dist['relative_freq']
        )
        ax2.set_title('Relative Frequency')
        
        # Cumulative frequency
        ax3.plot(
            range(len(self.freq_dist)),
            self.freq_dist['cumulative_freq'],
            marker='o'
        )
        ax3.set_title('Cumulative Frequency')
        
        # Density plot
        sns.kdeplot(self.data, ax=ax4)
        ax4.set_title('Density Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self) -> None:
        """Print frequency distribution summary"""
        print("\nFrequency Distribution Summary:")
        print(self.freq_dist.round(3))
        
        print("\nDistribution Statistics:")
        print(f"Number of bins: {self.bins}")
        print(f"Most common bin frequency: "
              f"{self.freq_dist['frequency'].max()}")
        print(f"Median frequency: "
              f"{self.freq_dist['frequency'].median()}")

# Example with student grades
grades = pd.Series(np.random.normal(75, 10, 200).clip(0, 100))
freq_analyzer = FrequencyAnalyzer(grades)
freq_analyzer.print_summary()
freq_analyzer.plot_distributions()
```
{% endstep %}

{% step %}
### Advanced Visualization Techniques
Let's create publication-quality visualizations:

```python
def create_analysis_dashboard(
    data: pd.Series,
    title: str = "Data Analysis Dashboard"
) -> None:
    """Create comprehensive data analysis dashboard"""
    # Setup the plot
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)
    
    # Basic distribution
    ax1 = fig.add_subplot(gs[0, :2])
    sns.histplot(data, kde=True, ax=ax1)
    ax1.set_title('Distribution with KDE')
    
    # Box plot
    ax2 = fig.add_subplot(gs[0, 2])
    sns.boxplot(y=data, ax=ax2)
    ax2.set_title('Box Plot')
    
    # QQ plot
    ax3 = fig.add_subplot(gs[1, 0])
    stats.probplot(data, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot')
    
    # Cumulative distribution
    ax4 = fig.add_subplot(gs[1, 1])
    stats.cumfreq(data, numbins=20)
    ax4.hist(
        data,
        bins=20,
        density=True,
        cumulative=True,
        alpha=0.8
    )
    ax4.set_title('Cumulative Distribution')
    
    # Violin plot
    ax5 = fig.add_subplot(gs[1, 2])
    sns.violinplot(y=data, ax=ax5)
    ax5.set_title('Violin Plot')
    
    # Add overall title
    fig.suptitle(title, size=16, y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(data.describe().round(2))
    
    # Print distribution tests
    print("\nNormality Tests:")
    _, shapiro_p = stats.shapiro(data)
    _, ks_p = stats.kstest(
        data,
        'norm',
        args=(data.mean(), data.std())
    )
    
    print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
    print(f"Kolmogorov-Smirnov p-value: {ks_p:.4f}")

# Example with real estate data
house_sizes = pd.Series(
    np.random.lognormal(7, 0.3, 200),  # Square feet
    name='House Sizes'
)

create_analysis_dashboard(
    house_sizes,
    "Real Estate Size Analysis"
)
```
{% endstep %}
{% endstepper %}

## Practice Exercises 🎯

Try these data analysis exercises:

1. **Analyze Customer Data**
   ```python
   # Create functions to:
   # - Load and clean customer data
   # - Calculate key statistics
   # - Identify customer segments
   # - Visualize distributions
   ```

2. **Financial Analysis**
   ```python
   # Build analysis tools for:
   # - Stock price distributions
   # - Return calculations
   # - Risk metrics
   # - Performance visualization
   ```

3. **Environmental Data**
   ```python
   # Analyze temperature data:
   # - Identify seasonal patterns
   # - Detect anomalies
   # - Calculate climate metrics
   # - Create time-based visualizations
   ```

Remember:
- Use appropriate statistical measures
- Create clear visualizations
- Handle outliers appropriately
- Document your analysis
- Consider the context of your data

Happy analyzing! 🚀
