import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def generate_quality_control_visualization():
    """Generate quality control visualization comparing population and sample distributions"""
    np.random.seed(42)
    # Simulate a production batch of 10,000 items
    population = np.random.normal(loc=100, scale=5, size=10000)
    # Take a sample of 100 items
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

    print("\nQuality Control Analysis")
    print(f"Population mean: {np.mean(population):.2f}")
    print(f"Sample mean: {np.mean(sample):.2f}")
    print(f"Difference: {abs(np.mean(population) - np.mean(sample)):.2f}")


def generate_sampling_methods_visualizations():
    """Generate visualizations for different sampling methods"""
    # Simple Random Sampling
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

    # Stratified Sampling
    population = np.arange(3000)
    strata_sizes = [1000, 1000, 1000]
    sample_sizes = [50, 50, 50]
    start_idx = 0
    colors = ["blue", "green", "red"]

    plt.figure(figsize=(12, 6))
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

    # Systematic Sampling
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

    # Cluster Sampling
    population = np.arange(1000)
    n_clusters = 5
    cluster_size = 20
    clusters = np.random.choice(
        len(population) // cluster_size, size=n_clusters, replace=False
    )

    plt.figure(figsize=(12, 6))
    plt.scatter(population, [0] * len(population), alpha=0.3, label="Population")

    for i, cluster in enumerate(clusters):
        start = cluster * cluster_size
        end = start + cluster_size
        cluster_members = population[start:end]

        plt.scatter(
            cluster_members,
            [0.1] * len(cluster_members),
            color=f"C{i}",
            label=f"Cluster {i+1}" if i == 0 else "",
        )

    plt.title(f"Cluster Sampling ({n_clusters} clusters of size {cluster_size})")
    plt.xlabel("Population ID")
    plt.yticks([])
    plt.legend()
    plt.savefig("cluster_sampling.png")
    plt.close()


def generate_sampling_error_visualization():
    """Generate visualization for sampling error vs sample size"""
    population_std = 15
    sample_sizes = [10, 100, 1000]
    ses = []

    plt.figure(figsize=(10, 6))
    for n in sample_sizes:
        se = population_std / np.sqrt(n)
        ses.append(se)
        print(f"Sample size {n}: Standard Error = {se:.2f}")

    plt.plot(sample_sizes, ses, "bo-")
    plt.xlabel("Sample Size")
    plt.ylabel("Standard Error")
    plt.title("Effect of Sample Size on Standard Error")
    plt.grid(True)
    plt.savefig("sampling_error_effect.png")
    plt.close()


def generate_sample_size_visualization():
    """Generate visualization for sample size determination"""
    confidence_level = 0.95
    p = 0.5
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    margins = np.linspace(0.01, 0.1, 100)
    sizes = [(z_score**2 * p * (1 - p)) / m**2 for m in margins]

    plt.figure(figsize=(10, 6))
    plt.plot(margins, sizes)
    plt.xlabel("Margin of Error")
    plt.ylabel("Required Sample Size")
    plt.title("Sample Size vs Margin of Error")
    plt.grid(True)
    plt.savefig("sample_size_relationship.png")
    plt.close()


if __name__ == "__main__":
    print("Generating population and sampling visualizations...")
    generate_quality_control_visualization()
    generate_sampling_methods_visualizations()
    generate_sampling_error_visualization()
    generate_sample_size_visualization()
    print("All population and sampling visualizations generated successfully!")
