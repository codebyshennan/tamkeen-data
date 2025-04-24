import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def generate_recovery_times_visualization():
    """Generate visualization for clinical trial example"""
    np.random.seed(42)

    # Control group (standard treatment)
    control = np.random.normal(loc=10, scale=2, size=30)  # Mean: 10 days

    # Treatment group (new medicine)
    treatment = np.random.normal(loc=9, scale=2, size=30)  # Mean: 9 days

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(control, treatment)

    # Visualize the distributions
    plt.figure(figsize=(10, 6))
    plt.hist(control, alpha=0.5, label="Control", bins=15)
    plt.hist(treatment, alpha=0.5, label="Treatment", bins=15)
    plt.axvline(np.mean(control), color="blue", linestyle="--", label="Control Mean")
    plt.axvline(
        np.mean(treatment), color="orange", linestyle="--", label="Treatment Mean"
    )
    plt.xlabel("Recovery Time (days)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Recovery Times")
    plt.legend()
    plt.savefig("recovery_times_distribution.png")
    plt.close()


def generate_effect_size_comparison():
    """Generate visualization for effect size vs p-value comparison"""
    np.random.seed(42)

    # Scenario 1: Small effect, large sample
    large_sample1 = np.random.normal(100, 10, 1000)
    large_sample2 = np.random.normal(101, 10, 1000)  # Just 1% difference

    # Scenario 2: Large effect, small sample
    small_sample1 = np.random.normal(100, 10, 20)
    small_sample2 = np.random.normal(110, 10, 20)  # 10% difference

    # Calculate p-values and effect sizes
    _, p_value1 = stats.ttest_ind(large_sample1, large_sample2)
    _, p_value2 = stats.ttest_ind(small_sample1, small_sample2)

    effect_size1 = (np.mean(large_sample2) - np.mean(large_sample1)) / np.std(
        large_sample1
    )
    effect_size2 = (np.mean(small_sample2) - np.mean(small_sample1)) / np.std(
        small_sample1
    )

    # Visualize the scenarios
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.hist(large_sample1, alpha=0.5, label="Group 1", bins=30)
    plt.hist(large_sample2, alpha=0.5, label="Group 2", bins=30)
    plt.title(f"Small Effect (p={p_value1:.4f})")
    plt.legend()

    plt.subplot(122)
    plt.hist(small_sample1, alpha=0.5, label="Group 1", bins=15)
    plt.hist(small_sample2, alpha=0.5, label="Group 2", bins=15)
    plt.title(f"Large Effect (p={p_value2:.4f})")
    plt.legend()

    plt.tight_layout()
    plt.savefig("effect_size_comparison.png")
    plt.close()


def generate_sample_size_effect():
    """Generate visualization for sample size effect on p-values"""
    np.random.seed(42)
    effect_size = 0.2  # Fixed small effect
    sizes = [20, 100, 500, 1000]

    # Visualize the effect of sample size
    plt.figure(figsize=(10, 6))
    for n in sizes:
        control = np.random.normal(0, 1, n)
        treatment = np.random.normal(effect_size, 1, n)
        _, p_value = stats.ttest_ind(control, treatment)

        plt.subplot(2, 2, sizes.index(n) + 1)
        plt.hist(control, alpha=0.5, label="Control", bins=15)
        plt.hist(treatment, alpha=0.5, label="Treatment", bins=15)
        plt.title(f"n={n}, p={p_value:.4f}")
        plt.legend()

    plt.tight_layout()
    plt.savefig("sample_size_effect.png")
    plt.close()


def generate_ab_test_visualization():
    """Generate visualization for A/B testing example"""
    np.random.seed(42)
    n_visitors = 1000

    # Control: 10% conversion rate
    # Treatment: 12% conversion rate
    control = np.random.binomial(1, 0.10, n_visitors)
    treatment = np.random.binomial(1, 0.12, n_visitors)

    # Create contingency table
    table = np.array(
        [
            [np.sum(control), len(control) - np.sum(control)],
            [np.sum(treatment), len(treatment) - np.sum(treatment)],
        ]
    )

    _, p_value, _, _ = stats.chi2_contingency(table)

    # Visualize the results
    plt.figure(figsize=(8, 6))
    plt.bar(
        ["Control", "Treatment"],
        [np.mean(control), np.mean(treatment)],
        yerr=[
            np.std(control) / np.sqrt(len(control)),
            np.std(treatment) / np.sqrt(len(treatment)),
        ],
        capsize=10,
    )
    plt.title(f"A/B Test Results (p={p_value:.4f})")
    plt.ylabel("Conversion Rate")
    plt.ylim(0, 0.2)
    plt.savefig("ab_test_results.png")
    plt.close()


def generate_multiple_testing_correction():
    """Generate visualization for multiple testing correction"""
    np.random.seed(42)

    # Simulate multiple tests
    p_values = [
        stats.ttest_ind(np.random.normal(0, 1, 30), np.random.normal(0, 1, 30)).pvalue
        for _ in range(20)
    ]

    # Apply Bonferroni correction
    corrected_p = multipletests(p_values, method="bonferroni")[1]

    # Visualize the correction
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(p_values)), p_values, label="Original p-values")
    plt.scatter(range(len(corrected_p)), corrected_p, label="Corrected p-values")
    plt.axhline(y=0.05, color="r", linestyle="--", label="α = 0.05")
    plt.xlabel("Test Number")
    plt.ylabel("P-value")
    plt.title("Multiple Testing Correction")
    plt.legend()
    plt.savefig("multiple_testing_correction.png")
    plt.close()


if __name__ == "__main__":
    print("Generating p-value visualizations...")
    generate_recovery_times_visualization()
    generate_effect_size_comparison()
    generate_sample_size_effect()
    generate_ab_test_visualization()
    generate_multiple_testing_correction()
    print("All p-value visualizations generated successfully!")
    print("All p-value visualizations generated successfully!")
