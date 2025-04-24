import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle, Rectangle
from scipy import stats

# Set style
plt.style.use("default")
sns.set_style("whitegrid")
sns.set_palette("husl")


def create_population_sample_diagram():
    """Create a diagram showing population and sample relationship"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create population circle
    population = Circle((0.5, 0.5), 0.4, fill=True, color="lightblue", alpha=0.5)
    ax.add_patch(population)

    # Create sample circle
    sample = Circle((0.5, 0.5), 0.2, fill=True, color="red", alpha=0.5)
    ax.add_patch(sample)

    # Add dots representing individuals
    np.random.seed(42)
    pop_points = np.random.uniform(0.1, 0.9, (200, 2))
    sample_points = np.random.uniform(0.3, 0.7, (50, 2))

    # Plot points
    ax.scatter(pop_points[:, 0], pop_points[:, 1], s=10, color="blue", alpha=0.5)
    ax.scatter(sample_points[:, 0], sample_points[:, 1], s=10, color="red", alpha=0.7)

    # Add labels
    ax.text(0.5, 0.5, "Population", ha="center", va="center", fontsize=12)
    ax.text(0.5, 0.5, "Sample", ha="center", va="center", fontsize=12, color="white")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    plt.savefig("population_sample_diagram.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_sampling_methods_diagram():
    """Create diagrams for different sampling methods"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Simple Random Sampling
    np.random.seed(42)
    points = np.random.uniform(0, 1, (100, 2))
    selected = np.random.choice(len(points), 20, replace=False)

    axes[0, 0].scatter(points[:, 0], points[:, 1], color="blue", alpha=0.3)
    axes[0, 0].scatter(points[selected, 0], points[selected, 1], color="red")
    axes[0, 0].set_title("Simple Random Sampling")
    axes[0, 0].axis("off")

    # Stratified Sampling
    strata1 = np.random.uniform(0, 0.5, (25, 2))
    strata2 = np.random.uniform(0.5, 1, (25, 2))
    selected1 = np.random.choice(len(strata1), 5, replace=False)
    selected2 = np.random.choice(len(strata2), 5, replace=False)

    axes[0, 1].scatter(strata1[:, 0], strata1[:, 1], color="blue", alpha=0.3)
    axes[0, 1].scatter(strata2[:, 0], strata2[:, 1], color="green", alpha=0.3)
    axes[0, 1].scatter(strata1[selected1, 0], strata1[selected1, 1], color="red")
    axes[0, 1].scatter(strata2[selected2, 0], strata2[selected2, 1], color="red")
    axes[0, 1].set_title("Stratified Sampling")
    axes[0, 1].axis("off")

    # Systematic Sampling
    points = np.array([(i, 0.5) for i in np.linspace(0, 1, 20)])
    selected = points[::2]

    axes[1, 0].scatter(points[:, 0], points[:, 1], color="blue", alpha=0.3)
    axes[1, 0].scatter(selected[:, 0], selected[:, 1], color="red")
    axes[1, 0].set_title("Systematic Sampling")
    axes[1, 0].axis("off")

    # Cluster Sampling
    clusters = []
    for i in range(4):
        center = np.random.uniform(0, 1, 2)
        cluster = center + np.random.normal(0, 0.05, (10, 2))
        clusters.append(cluster)

    selected_cluster = np.random.choice(4, 1)[0]

    for i, cluster in enumerate(clusters):
        color = "red" if i == selected_cluster else "blue"
        alpha = 1.0 if i == selected_cluster else 0.3
        axes[1, 1].scatter(cluster[:, 0], cluster[:, 1], color=color, alpha=alpha)

    axes[1, 1].set_title("Cluster Sampling")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("sampling_methods_diagram.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_sampling_error_diagram():
    """Create a diagram showing sampling error concept"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate population data
    np.random.seed(42)
    population = np.random.normal(100, 15, 10000)

    # Take multiple samples
    sample_means = []
    for _ in range(100):
        sample = np.random.choice(population, 100)
        sample_means.append(np.mean(sample))

    # Plot population distribution
    sns.kdeplot(population, ax=ax, label="Population", color="blue")

    # Plot sampling distribution
    sns.kdeplot(sample_means, ax=ax, label="Sampling Distribution", color="red")

    # Add vertical lines for means
    ax.axvline(
        np.mean(population), color="blue", linestyle="--", label="Population Mean"
    )
    ax.axvline(
        np.mean(sample_means), color="red", linestyle="--", label="Mean of Sample Means"
    )

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Sampling Error Visualization")
    ax.legend()

    plt.savefig("sampling_error_diagram.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_sample_size_effect_diagram():
    """Create a diagram showing the effect of sample size"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate population data
    np.random.seed(42)
    population = np.random.normal(100, 15, 10000)

    # Different sample sizes
    sample_sizes = [10, 30, 100, 300]
    colors = ["red", "green", "blue", "purple"]

    for size, color in zip(sample_sizes, colors):
        sample_means = []
        for _ in range(100):
            sample = np.random.choice(population, size)
            sample_means.append(np.mean(sample))

        sns.kdeplot(sample_means, ax=ax, label=f"n={size}", color=color)

    ax.set_xlabel("Sample Mean")
    ax.set_ylabel("Density")
    ax.set_title("Effect of Sample Size on Sampling Distribution")
    ax.legend()

    plt.savefig("sample_size_effect_diagram.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_parameter_statistic_diagram():
    """Create a diagram showing the relationship between parameters and statistics"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create population and sample boxes
    pop_box = Rectangle((0.1, 0.1), 0.4, 0.8, fill=True, color="lightblue", alpha=0.5)
    sample_box = Rectangle(
        (0.5, 0.1), 0.4, 0.8, fill=True, color="lightgreen", alpha=0.5
    )
    ax.add_patch(pop_box)
    ax.add_patch(sample_box)

    # Add labels
    ax.text(0.3, 0.9, "Population (Parameters)", ha="center", va="center", fontsize=12)
    ax.text(0.7, 0.9, "Sample (Statistics)", ha="center", va="center", fontsize=12)

    # Add parameter symbols
    param_symbols = ["μ", "σ", "σ²", "ρ", "π"]
    param_names = ["Mean", "Std Dev", "Variance", "Correlation", "Proportion"]
    for i, (sym, name) in enumerate(zip(param_symbols, param_names)):
        y_pos = 0.8 - i * 0.15
        ax.text(0.3, y_pos, f"{sym} = {name}", ha="center", va="center", fontsize=10)

    # Add statistic symbols
    stat_symbols = ["x̄", "s", "s²", "r", "p"]
    for i, sym in enumerate(stat_symbols):
        y_pos = 0.8 - i * 0.15
        ax.text(0.7, y_pos, sym, ha="center", va="center", fontsize=10)

    # Add arrows
    ax.arrow(
        0.5, 0.5, 0.1, 0, head_width=0.05, head_length=0.05, fc="black", ec="black"
    )
    ax.text(0.55, 0.5, "Estimate", ha="center", va="center", fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    plt.savefig("parameter_statistic_diagram.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_estimator_properties_diagram():
    """Create a diagram showing properties of good estimators"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Unbiasedness
    np.random.seed(42)
    true_value = 100
    n_simulations = 1000

    # Generate samples for unbiased estimator
    unbiased_means = []
    biased_means = []
    for _ in range(n_simulations):
        sample = np.random.normal(true_value, 15, 100)
        unbiased_means.append(np.mean(sample))
        biased_means.append(np.mean(sample) + 5)  # Add bias

    # Plot unbiasedness
    axes[0].hist(unbiased_means, bins=30, alpha=0.7, label="Unbiased")
    axes[0].hist(biased_means, bins=30, alpha=0.7, label="Biased")
    axes[0].axvline(true_value, color="red", linestyle="--", label="True Value")
    axes[0].set_title("Unbiasedness")
    axes[0].legend()

    # Efficiency
    efficient_means = []
    inefficient_means = []
    for _ in range(n_simulations):
        sample = np.random.normal(true_value, 15, 100)
        efficient_means.append(np.mean(sample))
        inefficient_means.append(stats.trim_mean(sample, 0.1))

    # Plot efficiency
    axes[1].hist(efficient_means, bins=30, alpha=0.7, label="Efficient")
    axes[1].hist(inefficient_means, bins=30, alpha=0.7, label="Less Efficient")
    axes[1].set_title("Efficiency")
    axes[1].legend()

    # Consistency
    sample_sizes = [10, 50, 100, 500, 1000]
    mean_diffs = []
    for size in sample_sizes:
        sample_means = []
        for _ in range(100):
            sample = np.random.normal(true_value, 15, size)
            sample_means.append(np.mean(sample))
        mean_diffs.append(np.mean([abs(m - true_value) for m in sample_means]))

    # Plot consistency
    axes[2].plot(sample_sizes, mean_diffs, marker="o")
    axes[2].set_title("Consistency")
    axes[2].set_xlabel("Sample Size")
    axes[2].set_ylabel("Average Error")

    plt.tight_layout()
    plt.savefig("estimator_properties_diagram.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_confidence_interval_diagram():
    """Create a diagram showing confidence intervals"""
    fig, ax = plt.subplots(figsize=(10, 6))

    np.random.seed(42)
    true_mean = 100
    n_intervals = 20
    sample_size = 30

    # Generate confidence intervals
    intervals = []
    for _ in range(n_intervals):
        sample = np.random.normal(true_mean, 15, sample_size)
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)
        margin = stats.t.ppf(0.975, sample_size - 1) * (
            sample_std / np.sqrt(sample_size)
        )
        intervals.append((sample_mean - margin, sample_mean + margin))

    # Plot intervals
    for i, (lower, upper) in enumerate(intervals):
        color = "green" if lower <= true_mean <= upper else "red"
        ax.plot([lower, upper], [i, i], color=color, linewidth=2)
        ax.plot([(lower + upper) / 2], [i], "o", color=color)

    # Add true mean line
    ax.axvline(true_mean, color="blue", linestyle="--", label="True Mean")

    ax.set_yticks(range(n_intervals))
    ax.set_yticklabels([f"Interval {i+1}" for i in range(n_intervals)])
    ax.set_xlabel("Value")
    ax.set_title("Confidence Intervals (95%)")
    ax.legend()

    plt.tight_layout()
    plt.savefig("confidence_interval_diagram.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_sampling_distribution_comparison():
    """Create a comparison of sampling distributions for different statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Generate population data
    np.random.seed(42)
    population = np.random.normal(100, 15, 10000)

    # Different statistics to compare
    statistics = {
        "Mean": lambda x: np.mean(x),
        "Median": lambda x: np.median(x),
        "Standard Deviation": lambda x: np.std(x, ddof=1),
        "Variance": lambda x: np.var(x, ddof=1),
    }

    # Generate sampling distributions
    n_samples = 1000
    sample_size = 30

    for (name, stat_func), ax in zip(statistics.items(), axes.flatten()):
        sample_stats = []
        for _ in range(n_samples):
            sample = np.random.choice(population, sample_size)
            sample_stats.append(stat_func(sample))

        # Plot sampling distribution
        sns.histplot(sample_stats, kde=True, ax=ax)
        ax.set_title(f"Sampling Distribution of {name}")
        ax.set_xlabel(f"Sample {name}")
        ax.set_ylabel("Frequency")

        # Add population parameter
        if name == "Mean":
            ax.axvline(
                np.mean(population),
                color="red",
                linestyle="--",
                label="Population Mean",
            )
        elif name == "Median":
            ax.axvline(
                np.median(population),
                color="red",
                linestyle="--",
                label="Population Median",
            )
        elif name == "Standard Deviation":
            ax.axvline(
                np.std(population, ddof=1),
                color="red",
                linestyle="--",
                label="Population Std Dev",
            )
        elif name == "Variance":
            ax.axvline(
                np.var(population, ddof=1),
                color="red",
                linestyle="--",
                label="Population Variance",
            )

        ax.legend()

    plt.tight_layout()
    plt.savefig("sampling_distribution_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_clt_visualization():
    """Create a comprehensive CLT visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Different population distributions
    distributions = {
        "Exponential": lambda size: np.random.exponential(scale=2, size=size),
        "Uniform": lambda size: np.random.uniform(0, 4, size),
        "Poisson": lambda size: np.random.poisson(lam=2, size=size),
        "Binomial": lambda size: np.random.binomial(n=10, p=0.3, size=size),
    }

    # Sample sizes to demonstrate
    sample_sizes = [5, 30, 100]
    n_samples = 1000

    for (name, dist_func), ax in zip(distributions.items(), axes.flatten()):
        # Generate population
        population = dist_func(10000)

        # Plot population distribution
        sns.histplot(population, kde=True, ax=ax, alpha=0.3, label="Population")

        # Plot sampling distributions for different sample sizes
        for size in sample_sizes:
            sample_means = []
            for _ in range(n_samples):
                sample = np.random.choice(population, size)
                sample_means.append(np.mean(sample))

            sns.kdeplot(sample_means, ax=ax, label=f"n={size}")

        ax.set_title(f"CLT Demonstration: {name} Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    plt.savefig("clt_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_standard_error_visualization():
    """Create a visualization of standard error concept"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Generate population
    np.random.seed(42)
    population = np.random.normal(100, 15, 10000)

    # Different sample sizes
    sample_sizes = [10, 30, 100, 300]
    n_samples = 100

    # Plot 1: Standard Error vs Sample Size
    ses = []
    for size in sample_sizes:
        sample_means = []
        for _ in range(n_samples):
            sample = np.random.choice(population, size)
            sample_means.append(np.mean(sample))
        ses.append(np.std(sample_means))

    axes[0].plot(sample_sizes, ses, "o-")
    axes[0].set_xlabel("Sample Size")
    axes[0].set_ylabel("Standard Error")
    axes[0].set_title("Standard Error vs Sample Size")

    # Plot 2: Confidence Intervals
    for i, size in enumerate(sample_sizes):
        sample = np.random.choice(population, size)
        sample_mean = np.mean(sample)
        se = np.std(sample, ddof=1) / np.sqrt(size)
        ci = stats.t.interval(0.95, size - 1, loc=sample_mean, scale=se)

        axes[1].plot([ci[0], ci[1]], [i, i], "o-")
        axes[1].plot([sample_mean], [i], "o", color="red")

    axes[1].axvline(
        np.mean(population), color="black", linestyle="--", label="Population Mean"
    )
    axes[1].set_xlabel("Value")
    axes[1].set_yticks(range(len(sample_sizes)))
    axes[1].set_yticklabels([f"n={size}" for size in sample_sizes])
    axes[1].set_title("95% Confidence Intervals")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("standard_error_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_bootstrap_visualization():
    """Create a visualization of bootstrap method"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Original sample
    np.random.seed(42)
    sample = np.random.normal(100, 15, 30)

    # Bootstrap
    n_bootstrap = 1000
    bootstrap_means = []

    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(sample, len(sample), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    # Plot 1: Bootstrap Distribution
    sns.histplot(bootstrap_means, kde=True, ax=axes[0])
    axes[0].axvline(np.mean(sample), color="red", linestyle="--", label="Sample Mean")
    axes[0].set_title("Bootstrap Distribution of the Mean")
    axes[0].set_xlabel("Bootstrap Mean")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Plot 2: Confidence Intervals
    ci_percentiles = [2.5, 97.5]
    ci = np.percentile(bootstrap_means, ci_percentiles)

    axes[1].hist(bootstrap_means, bins=30, alpha=0.7)
    axes[1].axvline(np.mean(sample), color="red", linestyle="--", label="Sample Mean")
    axes[1].axvline(ci[0], color="green", linestyle="--", label="95% CI Lower")
    axes[1].axvline(ci[1], color="green", linestyle="--", label="95% CI Upper")
    axes[1].set_title("Bootstrap Confidence Interval")
    axes[1].set_xlabel("Bootstrap Mean")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("bootstrap_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_p_value_concept_diagram():
    """Create a diagram showing the concept of p-values"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate normal distribution
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x)

    # Plot distribution
    ax.plot(x, y, "b-", label="Null Distribution")

    # Add observed value and p-value area
    observed = 2.5
    ax.axvline(observed, color="r", linestyle="--", label="Observed Value")

    # Shade p-value area
    x_fill = np.linspace(observed, 4, 100)
    y_fill = stats.norm.pdf(x_fill)
    ax.fill_between(x_fill, y_fill, alpha=0.3, color="red", label="P-value Area")

    ax.set_xlabel("Test Statistic")
    ax.set_ylabel("Density")
    ax.set_title("P-value Concept")
    ax.legend()

    plt.savefig("p_value_concept_diagram.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_p_value_calculation_diagram():
    """Create a diagram showing p-value calculation"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate normal distribution
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x)

    # Plot distribution
    ax.plot(x, y, "b-", label="Null Distribution")

    # Add observed value
    observed = 2.5
    ax.axvline(observed, color="r", linestyle="--", label="Observed Value")
    ax.axvline(-observed, color="r", linestyle="--")

    # Shade p-value areas
    x_fill1 = np.linspace(observed, 4, 100)
    x_fill2 = np.linspace(-4, -observed, 100)
    y_fill1 = stats.norm.pdf(x_fill1)
    y_fill2 = stats.norm.pdf(x_fill2)
    ax.fill_between(x_fill1, y_fill1, alpha=0.3, color="red", label="P-value Area")
    ax.fill_between(x_fill2, y_fill2, alpha=0.3, color="red")

    ax.set_xlabel("Test Statistic")
    ax.set_ylabel("Density")
    ax.set_title("P-value Calculation")
    ax.legend()

    plt.savefig("p_value_calculation_diagram.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_hypothesis_testing_diagram():
    """Create a diagram showing hypothesis testing framework"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate normal distribution
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x)

    # Plot distribution
    ax.plot(x, y, "b-", label="Null Distribution")

    # Add critical value and rejection region
    critical = 1.96
    ax.axvline(critical, color="r", linestyle="--", label="Critical Value")
    ax.axvline(-critical, color="r", linestyle="--")

    # Shade rejection regions
    x_fill1 = np.linspace(critical, 4, 100)
    x_fill2 = np.linspace(-4, -critical, 100)
    y_fill1 = stats.norm.pdf(x_fill1)
    y_fill2 = stats.norm.pdf(x_fill2)
    ax.fill_between(x_fill1, y_fill1, alpha=0.3, color="red", label="Rejection Region")
    ax.fill_between(x_fill2, y_fill2, alpha=0.3, color="red")

    ax.set_xlabel("Test Statistic")
    ax.set_ylabel("Density")
    ax.set_title("Hypothesis Testing Framework")
    ax.legend()

    plt.savefig("hypothesis_testing_diagram.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Create all visualizations
    print("Generating visualizations...")

    # Population and Sample
    create_population_sample_diagram()
    create_parameter_statistic_diagram()

    # Sampling Methods
    create_sampling_methods_diagram()
    create_sampling_error_diagram()
    create_sample_size_effect_diagram()

    # Confidence Intervals
    create_confidence_interval_diagram()

    # Sampling Distributions
    create_sampling_distribution_comparison()
    create_clt_visualization()
    create_standard_error_visualization()

    # P-values
    create_p_value_concept_diagram()
    create_p_value_calculation_diagram()
    create_hypothesis_testing_diagram()

    # Additional visualizations
    create_estimator_properties_diagram()
    create_bootstrap_visualization()

    print("All visualizations generated successfully!")
