import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def analyze_clinical_trial():
    """Generate treatment effects visualization"""
    np.random.seed(42)
    treatment_effect = np.random.normal(loc=10, scale=3, size=100)

    mean_effect = np.mean(treatment_effect)
    std_effect = np.std(treatment_effect, ddof=1)

    confidence = 0.95
    df = len(treatment_effect) - 1
    t_value = stats.t.ppf((1 + confidence) / 2, df)
    margin_error = t_value * (std_effect / np.sqrt(len(treatment_effect)))

    ci_lower = mean_effect - margin_error
    ci_upper = mean_effect + margin_error

    plt.figure(figsize=(10, 6))
    plt.hist(treatment_effect, bins=20, alpha=0.7, label="Treatment Effects")
    plt.axvline(mean_effect, color="red", linestyle="--", label="Mean Effect")
    plt.axvline(ci_lower, color="blue", linestyle=":", label="95% CI")
    plt.axvline(ci_upper, color="blue", linestyle=":")
    plt.fill_between(
        [ci_lower, ci_upper],
        [0, 0],
        [20, 20],
        color="blue",
        alpha=0.2,
        label="Confidence Interval",
    )
    plt.xlabel("Blood Pressure Reduction (mm Hg)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Treatment Effects with 95% CI")
    plt.legend()
    plt.savefig("treatment_effects_ci.png")
    plt.close()


def demonstrate_sample_size_effect():
    """Generate sample size effect visualization"""
    population_mean = 100
    population_std = 15
    sizes = [10, 30, 100, 300]

    plt.figure(figsize=(12, 8))
    for i, n in enumerate(sizes):
        plt.subplot(2, 2, i + 1)
        sample = np.random.normal(population_mean, population_std, n)
        ci = stats.t.interval(
            0.95, len(sample) - 1, loc=np.mean(sample), scale=stats.sem(sample)
        )

        plt.hist(sample, bins=15, alpha=0.7)
        plt.axvline(np.mean(sample), color="red", linestyle="--", label="Mean")
        plt.axvline(ci[0], color="blue", linestyle=":", label="95% CI")
        plt.axvline(ci[1], color="blue", linestyle=":")
        plt.fill_between([ci[0], ci[1]], [0, 0], [20, 20], color="blue", alpha=0.2)
        plt.title(f"Sample Size: {n}\nCI Width: {ci[1]-ci[0]:.2f}")
        plt.legend()

    plt.tight_layout()
    plt.savefig("sample_size_effect_ci.png")
    plt.close()


def demonstrate_confidence_level_effect():
    """Generate confidence level effect visualization"""
    sample = np.random.normal(100, 15, 30)
    levels = [0.80, 0.90, 0.95, 0.99]

    plt.figure(figsize=(12, 8))
    for i, level in enumerate(levels):
        plt.subplot(2, 2, i + 1)
        ci = stats.t.interval(
            level, len(sample) - 1, loc=np.mean(sample), scale=stats.sem(sample)
        )

        plt.hist(sample, bins=15, alpha=0.7)
        plt.axvline(np.mean(sample), color="red", linestyle="--", label="Mean")
        plt.axvline(ci[0], color="blue", linestyle=":", label=f"{level*100}% CI")
        plt.axvline(ci[1], color="blue", linestyle=":")
        plt.fill_between([ci[0], ci[1]], [0, 0], [20, 20], color="blue", alpha=0.2)
        plt.title(f"Confidence Level: {level*100}%\nCI Width: {ci[1]-ci[0]:.2f}")
        plt.legend()

    plt.tight_layout()
    plt.savefig("confidence_level_effect.png")
    plt.close()


def generate_test_scores_ci():
    # Example: Student test scores
    np.random.seed(42)
    scores = np.random.normal(75, 10, 50)
    mean = np.mean(scores)
    sem = stats.sem(scores)
    ci = stats.t.interval(0.95, len(scores) - 1, mean, sem)

    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=15, alpha=0.7)
    plt.axvline(mean, color="red", linestyle="--", label="Mean")
    plt.axvline(ci[0], color="blue", linestyle=":", label="95% CI")
    plt.axvline(ci[1], color="blue", linestyle=":")
    plt.fill_between([ci[0], ci[1]], [0, 0], [20, 20], color="blue", alpha=0.2)
    plt.xlabel("Test Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Test Scores with 95% CI")
    plt.legend()
    plt.savefig("test_scores_ci.png")
    plt.close()


def generate_survey_response_ci():
    # Example: Survey responses (1-5 scale)
    np.random.seed(42)
    responses = np.random.choice([1, 2, 3, 4, 5], size=100, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    mean = np.mean(responses)
    sem = stats.sem(responses)
    ci = stats.t.interval(0.95, len(responses) - 1, mean, sem)

    plt.figure(figsize=(10, 6))
    plt.hist(responses, bins=5, alpha=0.7, rwidth=0.8)
    plt.axvline(mean, color="red", linestyle="--", label="Mean")
    plt.axvline(ci[0], color="blue", linestyle=":", label="95% CI")
    plt.axvline(ci[1], color="blue", linestyle=":")
    plt.fill_between([ci[0], ci[1]], [0, 0], [50, 50], color="blue", alpha=0.2)
    plt.xlabel("Response Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Survey Responses with 95% CI")
    plt.legend()
    plt.savefig("survey_response_ci.png")
    plt.close()


def generate_teaching_methods_comparison():
    # Example: Comparing two teaching methods
    np.random.seed(42)
    method_a = np.random.normal(80, 10, 30)
    method_b = np.random.normal(85, 10, 30)

    plt.figure(figsize=(10, 6))
    plt.boxplot([method_a, method_b], labels=["Method A", "Method B"])
    plt.ylabel("Test Scores")
    plt.title("Comparison of Teaching Methods")
    plt.savefig("teaching_methods_comparison.png")
    plt.close()


if __name__ == "__main__":
    print("Generating confidence interval visualizations...")
    analyze_clinical_trial()
    demonstrate_sample_size_effect()
    demonstrate_confidence_level_effect()
    generate_test_scores_ci()
    generate_survey_response_ci()
    generate_teaching_methods_comparison()
    print("All CI visualizations generated successfully!")
