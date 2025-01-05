# Probability Distribution Families with Python 📊

## Understanding Distribution Families

{% stepper %}
{% step %}
### Distribution Families in Python
Let's explore different distribution families using Python:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Dict, Tuple, Optional

class DistributionFamilyExplorer:
    """Explore and analyze distribution families"""
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize explorer with optional seed"""
        if random_seed is not None:
            np.random.seed(random_seed)
        plt.style.use('seaborn')
    
    def plot_distribution_family(
        self,
        family: str,
        params_list: List[Dict[str, float]],
        n_samples: int = 1000
    ) -> None:
        """
        Plot distribution family with different parameters
        
        Args:
            family: Name of distribution family
            params_list: List of parameter dictionaries
            n_samples: Number of samples to generate
        """
        plt.figure(figsize=(12, 6))
        
        for params in params_list:
            if family == 'normal':
                data = np.random.normal(
                    loc=params['mean'],
                    scale=params['std'],
                    size=n_samples
                )
                label = f"μ={params['mean']}, σ={params['std']}"
            
            elif family == 'binomial':
                data = np.random.binomial(
                    n=params['n'],
                    p=params['p'],
                    size=n_samples
                )
                label = f"n={params['n']}, p={params['p']}"
            
            elif family == 'poisson':
                data = np.random.poisson(
                    lam=params['lambda'],
                    size=n_samples
                )
                label = f"λ={params['lambda']}"
            
            else:
                raise ValueError(f"Unknown family: {family}")
            
            if family == 'binomial':
                # For discrete distributions
                values, counts = np.unique(
                    data, return_counts=True
                )
                plt.bar(
                    values,
                    counts/n_samples,
                    alpha=0.5,
                    label=label
                )
            else:
                # For continuous distributions
                sns.kdeplot(data, label=label)
        
        plt.title(f'{family.title()} Distribution Family')
        plt.xlabel('Value')
        plt.ylabel('Density/Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# Example usage
explorer = DistributionFamilyExplorer(random_seed=42)

# Normal distribution family
normal_params = [
    {'mean': 0, 'std': 1},
    {'mean': 0, 'std': 2},
    {'mean': -2, 'std': 1.5}
]
explorer.plot_distribution_family('normal', normal_params)

# Binomial distribution family
binomial_params = [
    {'n': 10, 'p': 0.5},
    {'n': 20, 'p': 0.3},
    {'n': 15, 'p': 0.7}
]
explorer.plot_distribution_family('binomial', binomial_params)

# Poisson distribution family
poisson_params = [
    {'lambda': 2},
    {'lambda': 5},
    {'lambda': 8}
]
explorer.plot_distribution_family('poisson', poisson_params)
```
{% endstep %}

{% step %}
### Distribution Fitting and Testing
Let's create tools for fitting distributions to data:

```python
class DistributionFitter:
    """Fit and test probability distributions"""
    
    def __init__(self):
        """Initialize distribution families to test"""
        self.distributions = [
            stats.norm,
            stats.expon,
            stats.gamma,
            stats.lognorm,
            stats.weibull_min
        ]
    
    def fit_distribution(
        self,
        data: np.ndarray,
        dist: stats.rv_continuous
    ) -> Tuple[float, np.ndarray]:
        """
        Fit distribution and calculate goodness of fit
        
        Args:
            data: Input data
            dist: Distribution to fit
            
        Returns:
            Tuple of (p-value, parameters)
        """
        # Fit distribution
        params = dist.fit(data)
        
        # Perform Kolmogorov-Smirnov test
        _, p_value = stats.kstest(
            data,
            dist.name,
            params
        )
        
        return p_value, params
    
    def find_best_fit(
        self,
        data: np.ndarray
    ) -> Dict[str, Any]:
        """
        Find best fitting distribution
        
        Args:
            data: Input data
            
        Returns:
            Dictionary with best fit results
        """
        results = []
        
        for dist in self.distributions:
            try:
                p_value, params = self.fit_distribution(
                    data, dist
                )
                results.append({
                    'distribution': dist,
                    'p_value': p_value,
                    'params': params
                })
            except Exception as e:
                print(f"Error fitting {dist.name}: {str(e)}")
        
        # Sort by p-value
        results.sort(key=lambda x: x['p_value'], reverse=True)
        return results[0]
    
    def plot_fit_comparison(
        self,
        data: np.ndarray,
        fitted_dist: Dict[str, Any]
    ) -> None:
        """Plot data with fitted distribution"""
        plt.figure(figsize=(12, 6))
        
        # Plot histogram of data
        sns.histplot(
            data,
            stat='density',
            alpha=0.5,
            label='Data'
        )
        
        # Plot fitted distribution
        x = np.linspace(min(data), max(data), 100)
        dist = fitted_dist['distribution']
        params = fitted_dist['params']
        y = dist.pdf(x, *params)
        
        plt.plot(x, y, 'r-', lw=2,
                label=f'Fitted {dist.name}')
        
        plt.title('Data vs Fitted Distribution')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Print fit statistics
        print("\nFit Statistics:")
        print(f"Best fit distribution: {dist.name}")
        print(f"P-value: {fitted_dist['p_value']:.4f}")
        print("\nParameters:")
        for i, param in enumerate(params):
            print(f"Parameter {i+1}: {param:.4f}")

# Example usage
fitter = DistributionFitter()

# Generate sample data
np.random.seed(42)
data = np.random.lognormal(mean=0, sigma=0.5, size=1000)

# Find and plot best fit
best_fit = fitter.find_best_fit(data)
fitter.plot_fit_comparison(data, best_fit)
```
{% endstep %}
{% endstepper %}

## Common Distribution Families

{% stepper %}
{% step %}
### Binomial Distribution
Let's implement tools for working with binomial distributions:

```python
class BinomialAnalyzer:
    """Analyze binomial distributions"""
    
    def __init__(
        self,
        n: int,
        p: float,
        random_seed: Optional[int] = None
    ):
        """
        Initialize analyzer
        
        Args:
            n: Number of trials
            p: Probability of success
            random_seed: Random seed
        """
        self.n = n
        self.p = p
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def simulate_trials(
        self,
        n_simulations: int = 1000
    ) -> pd.Series:
        """Simulate binomial trials"""
        return pd.Series(
            np.random.binomial(self.n, self.p, n_simulations),
            name='Successes'
        )
    
    def calculate_probability(
        self,
        k: int
    ) -> float:
        """Calculate probability of exactly k successes"""
        return stats.binom.pmf(k, self.n, self.p)
    
    def plot_distribution(
        self,
        data: Optional[pd.Series] = None
    ) -> None:
        """Plot binomial distribution"""
        plt.figure(figsize=(12, 6))
        
        # Plot theoretical probabilities
        x = np.arange(0, self.n + 1)
        pmf = stats.binom.pmf(x, self.n, self.p)
        plt.bar(x, pmf, alpha=0.5, label='Theoretical')
        
        # Plot simulated data if provided
        if data is not None:
            value_counts = data.value_counts(normalize=True)
            plt.bar(value_counts.index, value_counts.values,
                   alpha=0.5, label='Simulated')
        
        plt.title('Binomial Distribution')
        plt.xlabel('Number of Successes')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Print statistics
        print("\nDistribution Statistics:")
        print(f"Mean: {self.n * self.p:.2f}")
        print(f"Variance: {self.n * self.p * (1-self.p):.2f}")
        if data is not None:
            print("\nSimulated Statistics:")
            print(data.describe().round(2))

# Example usage
analyzer = BinomialAnalyzer(n=10, p=0.3, random_seed=42)

# Simulate trials
simulated_data = analyzer.simulate_trials(10000)

# Plot distribution
analyzer.plot_distribution(simulated_data)

# Calculate specific probability
prob_5 = analyzer.calculate_probability(5)
print(f"\nProbability of exactly 5 successes: {prob_5:.4f}")
```
{% endstep %}

{% step %}
### Poisson Distribution
Implementation for Poisson distributions:

```python
class PoissonAnalyzer:
    """Analyze Poisson distributions"""
    
    def __init__(
        self,
        lambda_: float,
        random_seed: Optional[int] = None
    ):
        """
        Initialize analyzer
        
        Args:
            lambda_: Average rate
            random_seed: Random seed
        """
        self.lambda_ = lambda_
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def simulate_events(
        self,
        n_simulations: int = 1000
    ) -> pd.Series:
        """Simulate Poisson events"""
        return pd.Series(
            np.random.poisson(self.lambda_, n_simulations),
            name='Events'
        )
    
    def calculate_probability(
        self,
        k: int
    ) -> float:
        """Calculate probability of exactly k events"""
        return stats.poisson.pmf(k, self.lambda_)
    
    def plot_distribution(
        self,
        data: Optional[pd.Series] = None,
        max_k: Optional[int] = None
    ) -> None:
        """Plot Poisson distribution"""
        if max_k is None:
            max_k = int(self.lambda_ * 3)
        
        plt.figure(figsize=(12, 6))
        
        # Plot theoretical probabilities
        x = np.arange(0, max_k + 1)
        pmf = stats.poisson.pmf(x, self.lambda_)
        plt.bar(x, pmf, alpha=0.5, label='Theoretical')
        
        # Plot simulated data if provided
        if data is not None:
            value_counts = data.value_counts(normalize=True)
            plt.bar(value_counts.index, value_counts.values,
                   alpha=0.5, label='Simulated')
        
        plt.title('Poisson Distribution')
        plt.xlabel('Number of Events')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Print statistics
        print("\nDistribution Statistics:")
        print(f"Mean: {self.lambda_:.2f}")
        print(f"Variance: {self.lambda_:.2f}")
        if data is not None:
            print("\nSimulated Statistics:")
            print(data.describe().round(2))

# Example usage
analyzer = PoissonAnalyzer(lambda_=3, random_seed=42)

# Simulate events
simulated_data = analyzer.simulate_events(10000)

# Plot distribution
analyzer.plot_distribution(simulated_data)

# Calculate specific probability
prob_5 = analyzer.calculate_probability(5)
print(f"\nProbability of exactly 5 events: {prob_5:.4f}")
```
{% endstep %}

{% step %}
### Normal Distribution and Central Limit Theorem
Let's demonstrate the Central Limit Theorem:

```python
class CLTDemonstrator:
    """Demonstrate Central Limit Theorem"""
    
    def __init__(self, random_seed: Optional[int] = None):
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate_sample_means(
        self,
        distribution: str,
        params: Dict[str, float],
        sample_size: int,
        n_samples: int
    ) -> np.ndarray:
        """
        Generate means of random samples
        
        Args:
            distribution: Base distribution
            params: Distribution parameters
            sample_size: Size of each sample
            n_samples: Number of samples
            
        Returns:
            Array of sample means
        """
        if distribution == 'uniform':
            data = np.random.uniform(
                params['low'],
                params['high'],
                (n_samples, sample_size)
            )
        elif distribution == 'exponential':
            data = np.random.exponential(
                params['scale'],
                (n_samples, sample_size)
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        return np.mean(data, axis=1)
    
    def plot_clt_demonstration(
        self,
        distribution: str,
        params: Dict[str, float],
        sample_sizes: List[int],
        n_samples: int = 1000
    ) -> None:
        """Plot CLT demonstration"""
        n_sizes = len(sample_sizes)
        fig, axes = plt.subplots(
            n_sizes, 2,
            figsize=(15, 5 * n_sizes)
        )
        
        for i, size in enumerate(sample_sizes):
            # Generate sample means
            means = self.generate_sample_means(
                distribution,
                params,
                size,
                n_samples
            )
            
            # Plot histogram
            sns.histplot(
                means,
                stat='density',
                kde=True,
                ax=axes[i, 0]
            )
            axes[i, 0].set_title(
                f'Distribution of Sample Means (n={size})'
            )
            
            # Q-Q plot
            stats.probplot(
                means,
                dist='norm',
                plot=axes[i, 1]
            )
            axes[i, 1].set_title(
                f'Q-Q Plot (n={size})'
            )
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics for each sample size
        print("\nSample Statistics:")
        for size in sample_sizes:
            means = self.generate_sample_means(
                distribution,
                params,
                size,
                n_samples
            )
            print(f"\nSample Size: {size}")
            print(f"Mean: {np.mean(means):.4f}")
            print(f"Std Dev: {np.std(means):.4f}")
            _, p_value = stats.normaltest(means)
            print(f"Normality Test p-value: {p_value:.4f}")

# Example usage
demonstrator = CLTDemonstrator(random_seed=42)

# Demonstrate CLT with uniform distribution
params = {'low': 0, 'high': 1}
sample_sizes = [1, 5, 30, 100]

print("\nCLT Demonstration (Uniform Distribution):")
demonstrator.plot_clt_demonstration(
    'uniform',
    params,
    sample_sizes
)

# Demonstrate CLT with exponential distribution
params = {'scale': 2}
print("\nCLT Demonstration (Exponential Distribution):")
demonstrator.plot_clt_demonstration(
    'exponential',
    params,
    sample_sizes
)
```
{% endstep %}
{% endstepper %}

## Practice Exercises 🎯

Try these distribution analysis exercises:

1. **Customer Service Analysis**
   ```python
   # Create functions to:
   # - Analyze call arrival patterns
   # - Fit appropriate distribution
   # - Calculate staffing requirements
   ```

2. **Manufacturing Quality Control**
   ```python
   # Build tools to:
   # - Model defect rates
   # - Calculate control limits
   # - Predict batch quality
   ```

3. **Financial Risk Analysis**
   ```python
   # Implement system to:
   # - Analyze return distributions
   # - Calculate Value at Risk
   # - Model portfolio risk
   ```

Remember:
- Choose appropriate distribution family
- Validate distribution assumptions
- Consider sample size effects
- Use visualization for insights
- Document your analysis

Happy analyzing! 🚀
