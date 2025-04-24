import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def generate_population_sample_visualizations():
    """Generate visualizations for population-sample.md"""
    # Population and sample distribution
    population = np.random.normal(loc=100, scale=5, size=10000)
    sample = np.random.choice(population, size=100, replace=False)

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.hist(population, bins=30, alpha=0.7, color="blue")
    plt.axvline(
        np.mean(population), color="red", linestyle="--", label="Population Mean"
    )
    plt.title("Population Distribution")
    plt.xlabel("Measurement")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(122)
    plt.hist(sample, bins=15, alpha=0.7, color="green")
    plt.axvline(np.mean(sample), color="red", linestyle="--", label="Sample Mean")
    plt.title("Sample Distribution")
    plt.xlabel("Measurement")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.savefig("population_sample_dist.png")
    plt.close()

    # Simple random sampling
    population = np.arange(1000)
    sample = np.random.choice(population, size=100, replace=False)

    plt.figure(figsize=(10, 6))
    plt.scatter(population, [0] * len(population), alpha=0.3, label="Population")
    plt.scatter(sample, [0.1] * len(sample), color="red", label="Selected Sample")
    plt.title("Simple Random Sampling")
    plt.xlabel("Population ID")
    plt.yticks([])
    plt.legend()
    plt.savefig("simple_random_sampling.png")
    plt.close()

    # Stratified sampling
    population = np.arange(3000)
    strata_sizes = [1000, 1000, 1000]
    sample_sizes = [50, 50, 50]

    plt.figure(figsize=(12, 6))
    colors = ["blue", "green", "red"]
    start_idx = 0

    for i, (stratum_size, sample_size) in enumerate(zip(strata_sizes, sample_sizes)):
        stratum = population[start_idx : start_idx + stratum_size]
        sample = np.random.choice(stratum, size=sample_size, replace=False)

        plt.scatter(
            stratum,
            [i] * len(stratum),
            alpha=0.3,
            color=colors[i],
            label=f"Stratum {i+1}",
        )
        plt.scatter(
            sample,
            [i + 0.1] * len(sample),
            color=colors[i],
            label=f"Sample {i+1}" if i == 0 else "",
        )

        start_idx += stratum_size

    plt.title("Stratified Sampling")
    plt.xlabel("Population ID")
    plt.yticks(
        range(len(strata_sizes)), [f"Stratum {i+1}" for i in range(len(strata_sizes))]
    )
    plt.legend()
    plt.savefig("stratified_sampling.png")
    plt.close()

    # Systematic sampling
    population = np.arange(1000)
    interval = 10
    start = np.random.randint(0, interval)
    sample = population[start::interval]

    plt.figure(figsize=(10, 6))
    plt.scatter(population, [0] * len(population), alpha=0.3, label="Population")
    plt.scatter(sample, [0.1] * len(sample), color="red", label="Selected Sample")
    plt.title(f"Systematic Sampling (Interval: {interval})")
    plt.xlabel("Population ID")
    plt.yticks([])
    plt.legend()
    plt.savefig("systematic_sampling.png")
    plt.close()

    # Cluster sampling
    population = np.arange(1000)
    n_clusters = 5
    cluster_size = 20

    plt.figure(figsize=(12, 6))
    plt.scatter(population, [0] * len(population), alpha=0.3, label="Population")

    for i in range(n_clusters):
        cluster = np.random.choice(len(population), size=cluster_size, replace=False)
        plt.scatter(
            population[cluster],
            [0.1] * len(cluster),
            color=f"C{i}",
            label=f"Cluster {i+1}" if i == 0 else "",
        )

    plt.title(f"Cluster Sampling ({n_clusters} clusters of size {cluster_size})")
    plt.xlabel("Population ID")
    plt.yticks([])
    plt.legend()
    plt.savefig("cluster_sampling.png")
    plt.close()


def generate_sampling_distribution_visualizations():
    """Generate visualizations for sampling-distributions.md"""
    # CLT visualizations
    distributions = ["exponential", "uniform", "skewed"]
    for dist in distributions:
        plt.figure(figsize=(15, 5))

        if dist == "exponential":
            population = np.random.exponential(scale=1.0, size=10000)
            title = "Exponential Distribution"
        elif dist == "uniform":
            population = np.random.uniform(0, 1, 10000)
            title = "Uniform Distribution"
        else:  # Skewed custom distribution
            population = np.concatenate(
                [np.random.normal(0, 1, 7000), np.random.normal(3, 0.5, 3000)]
            )
            title = "Skewed Distribution"

        sample_size = 30
        n_samples = 1000
        sample_means = [
            np.mean(np.random.choice(population, size=sample_size))
            for _ in range(n_samples)
        ]

        plt.subplot(131)
        plt.hist(population, bins=50, density=True, alpha=0.7, color="skyblue")
        plt.title(f"Population Distribution\n({title})")
        plt.xlabel("Value")
        plt.ylabel("Density")

        plt.subplot(132)
        sample = np.random.choice(population, size=sample_size)
        plt.hist(sample, bins=20, density=True, alpha=0.7, color="lightgreen")
        plt.title(f"One Sample Distribution\n(n={sample_size})")
        plt.xlabel("Value")

        plt.subplot(133)
        plt.hist(sample_means, bins=30, density=True, alpha=0.7, color="salmon")
        x = np.linspace(min(sample_means), max(sample_means), 100)
        plt.plot(
            x,
            stats.norm.pdf(x, np.mean(sample_means), np.std(sample_means)),
            "k--",
            label="Normal Curve",
        )
        plt.title("Sampling Distribution\nof the Mean")
        plt.xlabel("Sample Mean")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"clt_{dist}.png")
        plt.close()

    # Standard error visualization
    population = np.random.normal(100, 15, 10000)
    sizes = [10, 30, 100, 300, 1000]

    plt.figure(figsize=(12, 8))
    for i, n in enumerate(sizes):
        plt.subplot(2, 3, i + 1)

        sample_means = [
            np.mean(np.random.choice(population, size=n)) for _ in range(1000)
        ]

        plt.hist(sample_means, bins=30, density=True, alpha=0.7)
        plt.axvline(
            np.mean(population), color="red", linestyle="--", label="Population Mean"
        )

        x = np.linspace(min(sample_means), max(sample_means), 100)
        plt.plot(
            x,
            stats.norm.pdf(x, np.mean(sample_means), np.std(sample_means)),
            "k--",
            label="Normal Curve",
        )

        plt.title(f"Sample Size: {n}\nSE: {np.std(sample_means):.2f}")
        plt.legend()

    plt.tight_layout()
    plt.savefig("standard_error_effect.png")
    plt.close()


def generate_p_value_visualizations():
    """Generate visualizations for p-values.md"""
    # Recovery times distribution
    control = np.random.normal(loc=10, scale=2, size=30)
    treatment = np.random.normal(loc=9, scale=2, size=30)

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

    # Effect size comparison
    large_sample1 = np.random.normal(100, 10, 1000)
    large_sample2 = np.random.normal(101, 10, 1000)
    small_sample1 = np.random.normal(100, 10, 20)
    small_sample2 = np.random.normal(110, 10, 20)

    _, p_value1 = stats.ttest_ind(large_sample1, large_sample2)
    _, p_value2 = stats.ttest_ind(small_sample1, small_sample2)

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

    # Sample size effect
    effect_size = 0.2
    sizes = [20, 100, 500, 1000]

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


if __name__ == "__main__":
    print("Generating all visualizations...")
    generate_population_sample_visualizations()
    generate_sampling_distribution_visualizations()
    generate_p_value_visualizations()
    print("All visualizations generated successfully!")
