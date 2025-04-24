import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def generate_tree_height_visualization():
    """Generate visualization for tree height example"""
    np.random.seed(42)
    # Simulate a population of tree heights (in feet)
    population = np.random.normal(loc=100, scale=15, size=10000)
    population_mean = np.mean(population)

    # Take a sample and calculate point estimate
    sample = np.random.choice(population, size=100)
    sample_mean = np.mean(sample)

    # Calculate 95% confidence interval
    confidence_level = 0.95
    sample_std = np.std(sample, ddof=1)
    margin_of_error = stats.t.ppf((1 + confidence_level) / 2, len(sample) - 1) * (
        sample_std / np.sqrt(len(sample))
    )
    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error

    # Visualize
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.hist(population, bins=50, alpha=0.7, color="blue", label="Population")
    plt.axvline(population_mean, color="red", linestyle="--", label="Population Mean")
    plt.title("Population Distribution")
    plt.xlabel("Tree Height (feet)")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(122)
    plt.hist(sample, bins=20, alpha=0.7, color="green", label="Sample")
    plt.axvline(sample_mean, color="red", linestyle="--", label="Sample Mean")
    plt.axvline(ci_lower, color="blue", linestyle=":", label="95% CI")
    plt.axvline(ci_upper, color="blue", linestyle=":")
    plt.fill_between([ci_lower, ci_upper], [0, 0], [20, 20], color="blue", alpha=0.2)
    plt.title("Sample Distribution with 95% CI")
    plt.xlabel("Tree Height (feet)")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.savefig("tree_height_analysis.png")
    plt.close()


def generate_unbiasedness_visualization():
    """Generate visualization for unbiasedness demonstration"""
    np.random.seed(42)
    population = np.random.normal(100, 15, 10000)
    population_mean = np.mean(population)

    # Simulate multiple samples
    n_simulations = 1000
    sample_means = []

    for _ in range(n_simulations):
        sample = np.random.choice(population, size=100)
        sample_means.append(np.mean(sample))

    # Visualize
    plt.figure(figsize=(10, 6))
    plt.hist(sample_means, bins=30, alpha=0.7, label="Sample Means")
    plt.axvline(population_mean, color="red", linestyle="--", label="Population Mean")
    plt.axvline(
        np.mean(sample_means),
        color="green",
        linestyle="--",
        label="Mean of Sample Means",
    )
    plt.title("Distribution of Sample Means")
    plt.xlabel("Sample Mean")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("unbiasedness_demonstration.png")
    plt.close()


def generate_efficiency_visualization():
    """Generate visualization for efficiency comparison"""
    np.random.seed(42)
    data = np.random.normal(100, 15, 1000)

    # Calculate different estimators
    regular_mean = np.mean(data)
    trimmed_mean = stats.trim_mean(data, 0.1)

    # Visualize
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, alpha=0.7, label="Data")
    plt.axvline(regular_mean, color="red", linestyle="--", label="Regular Mean")
    plt.axvline(trimmed_mean, color="green", linestyle="--", label="Trimmed Mean")
    plt.title("Comparison of Mean Estimators")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("efficiency_comparison.png")
    plt.close()


def generate_consistency_visualization():
    """Generate visualization for consistency demonstration"""
    np.random.seed(42)
    population = np.random.normal(100, 15, 10000)
    population_mean = np.mean(population)

    # Different sample sizes
    sample_sizes = [10, 100, 1000, 5000]
    results = []

    for size in sample_sizes:
        sample = np.random.choice(population, size=size)
        results.append(np.mean(sample))

    # Visualize
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, results, "bo-", label="Sample Means")
    plt.axhline(population_mean, color="red", linestyle="--", label="Population Mean")
    plt.title("Consistency of Sample Mean")
    plt.xlabel("Sample Size")
    plt.ylabel("Sample Mean")
    plt.legend()
    plt.grid(True)
    plt.savefig("consistency_demonstration.png")
    plt.close()


def generate_quality_control_visualization():
    """Generate visualization for quality control example"""
    np.random.seed(42)
    population_mean = 100
    population_std = 2
    sample_size = 30

    # Generate sample
    sample = np.random.normal(population_mean, population_std, sample_size)

    # Calculate statistics
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)

    # Calculate 95% confidence interval
    ci = stats.t.interval(
        0.95, len(sample) - 1, loc=sample_mean, scale=stats.sem(sample)
    )

    # Visualize
    plt.figure(figsize=(10, 6))
    plt.hist(sample, bins=15, alpha=0.7, label="Measurements")
    plt.axvline(population_mean, color="red", linestyle="--", label="Target Value")
    plt.axvline(sample_mean, color="green", linestyle="--", label="Sample Mean")
    plt.axvline(ci[0], color="blue", linestyle=":", label="95% CI")
    plt.axvline(ci[1], color="blue", linestyle=":")
    plt.fill_between([ci[0], ci[1]], [0, 0], [10, 10], color="blue", alpha=0.2)
    plt.title("Quality Control Measurements")
    plt.xlabel("Measurement Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("quality_control_analysis.png")
    plt.close()


def generate_ab_testing_visualization():
    """Generate visualization for A/B testing example"""
    np.random.seed(42)
    control_rate = 0.1
    treatment_rate = 0.12
    sample_size = 1000

    # Generate samples
    control = np.random.binomial(1, control_rate, sample_size)
    treatment = np.random.binomial(1, treatment_rate, sample_size)

    # Calculate statistics
    control_mean = np.mean(control)
    treatment_mean = np.mean(treatment)

    # Calculate difference and confidence interval
    diff = treatment_mean - control_mean
    diff_std = np.sqrt(
        control_mean * (1 - control_mean) / sample_size
        + treatment_mean * (1 - treatment_rate) / sample_size
    )
    ci = stats.norm.interval(0.95, loc=diff, scale=diff_std)

    # Visualize
    plt.figure(figsize=(10, 6))
    plt.bar(
        ["Control", "Treatment"],
        [control_mean, treatment_mean],
        yerr=[
            np.std(control) / np.sqrt(len(control)),
            np.std(treatment) / np.sqrt(len(treatment)),
        ],
        capsize=10,
    )
    plt.title("A/B Test Results")
    plt.ylabel("Conversion Rate")
    plt.ylim(0, 0.2)
    plt.savefig("ab_testing_results.png")
    plt.close()


if __name__ == "__main__":
    print("Generating parameter and statistics visualizations...")
    generate_tree_height_visualization()
    generate_unbiasedness_visualization()
    generate_efficiency_visualization()
    generate_consistency_visualization()
    generate_quality_control_visualization()
    generate_ab_testing_visualization()
    print("All parameter and statistics visualizations generated successfully!")
