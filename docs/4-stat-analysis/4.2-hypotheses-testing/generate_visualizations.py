import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from scipy import stats

# Set style
plt.style.use("ggplot")  # Using ggplot style instead of seaborn

# Ensure assets directory exists
os.makedirs("assets", exist_ok=True)


def save_figure(filename):
    """Helper function to save figures to the assets directory"""
    plt.savefig(
        os.path.join("assets", filename),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()


def create_scientific_method_flowchart():
    """Create a flowchart illustrating the scientific method"""
    G = nx.DiGraph()

    # Add nodes
    nodes = [
        "Question/Problem",
        "Research",
        "Hypothesis",
        "Experiment",
        "Analysis",
        "Conclusion",
        "Communication",
    ]

    # Add edges to create the flow
    edges = [
        ("Question/Problem", "Research"),
        ("Research", "Hypothesis"),
        ("Hypothesis", "Experiment"),
        ("Experiment", "Analysis"),
        ("Analysis", "Conclusion"),
        ("Conclusion", "Communication"),
        ("Conclusion", "Question/Problem"),  # Feedback loop
    ]

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Set up the plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Draw the graph
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=2000,
        font_size=10,
        font_weight="bold",
        arrows=True,
        edge_color="gray",
        arrowsize=20,
    )

    save_figure("scientific_method.png")


def create_statistical_test_decision_tree():
    """Create a decision tree for choosing statistical tests"""
    G = nx.DiGraph()

    # Add nodes
    nodes = {
        "Start": "Data Type?",
        "Numeric": "How many groups?",
        "Categorical": "How many variables?",
        "One": "One-sample t-test",
        "Two": "Paired or Independent?",
        "Many": "One-way ANOVA",
        "Single": "Chi-square test",
        "Multiple": "Chi-square independence",
    }

    # Add edges
    edges = [
        ("Start", "Numeric"),
        ("Start", "Categorical"),
        ("Numeric", "One"),
        ("Numeric", "Two"),
        ("Numeric", "Many"),
        ("Categorical", "Single"),
        ("Categorical", "Multiple"),
    ]

    # Create graph
    G.add_nodes_from(nodes.keys())
    G.add_edges_from(edges)

    # Set up plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=2000)
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20)

    # Add labels
    labels = {node: text for node, text in nodes.items()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold")

    save_figure("statistical_test_tree.png")


def create_ab_testing_timeline():
    """Create a timeline visualization for A/B testing process"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define stages and their positions
    stages = ["Planning", "Setup", "Data Collection", "Analysis", "Decision"]

    # Create timeline
    y_pos = 0
    for i, stage in enumerate(stages):
        ax.plot([i, i + 1], [y_pos, y_pos], "b-", linewidth=2)
        ax.plot([i + 0.5], [y_pos], "bo", markersize=10)
        ax.text(i + 0.5, y_pos + 0.1, stage, ha="center", va="bottom")

    # Customize plot
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(-0.2, len(stages))
    ax.axis("off")

    save_figure("ab_testing_timeline.png")


def create_effect_size_visualization():
    """Create visualization for different effect sizes"""
    np.random.seed(42)

    # Generate sample data
    control = np.random.normal(0, 1, 1000)
    small_effect = np.random.normal(0.2, 1, 1000)
    medium_effect = np.random.normal(0.5, 1, 1000)
    large_effect = np.random.normal(0.8, 1, 1000)

    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Plot distributions
    for ax, (data, title) in zip(
        axes,
        [
            (small_effect, "Small Effect (d=0.2)"),
            (medium_effect, "Medium Effect (d=0.5)"),
            (large_effect, "Large Effect (d=0.8)"),
        ],
    ):
        sns.kdeplot(data=control, ax=ax, label="Control")
        sns.kdeplot(data=data, ax=ax, label="Treatment")
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    save_figure("effect_sizes.png")


def create_confidence_interval_diagram():
    """Create visualization for confidence intervals"""
    np.random.seed(42)

    # Generate sample data
    mean = 0
    std = 1
    n = 100
    data = np.random.normal(mean, std, n)

    # Calculate confidence intervals
    confidence_levels = [0.68, 0.95, 0.99]
    z_scores = stats.norm.ppf([(1 + cl) / 2 for cl in confidence_levels])
    intervals = [
        (mean - z * std / np.sqrt(n), mean + z * std / np.sqrt(n)) for z in z_scores
    ]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot intervals
    y_positions = range(len(confidence_levels))
    for i, ((lower, upper), cl) in enumerate(zip(intervals, confidence_levels)):
        ax.plot([lower, upper], [i, i], "b-", linewidth=2)
        ax.plot([lower], [i], "b|", markersize=10)
        ax.plot([upper], [i], "b|", markersize=10)
        ax.text(upper + 0.1, i, f"{int(cl*100)}% CI", va="center")

    ax.axvline(x=mean, color="r", linestyle="--", label="True Mean")
    ax.set_yticks([])
    ax.set_xlabel("Value")
    ax.set_title("Confidence Intervals")

    save_figure("confidence_intervals.png")


def create_power_analysis_visualization():
    """Create visualization for power analysis"""
    # Generate sample sizes
    n = np.linspace(20, 200, 100)

    # Calculate power for different effect sizes
    effect_sizes = [0.2, 0.5, 0.8]  # Small, medium, large
    alpha = 0.05

    plt.figure(figsize=(10, 6))

    for d in effect_sizes:
        # Calculate power
        power_analysis = []
        for sample_size in n:
            # Calculate non-centrality parameter
            nc = d * np.sqrt(sample_size / 2)
            # Calculate power
            power_val = (
                1
                - stats.t.cdf(
                    stats.t.ppf(1 - alpha / 2, 2 * sample_size - 2),
                    2 * sample_size - 2,
                    nc,
                )
                + stats.t.cdf(
                    stats.t.ppf(alpha / 2, 2 * sample_size - 2), 2 * sample_size - 2, nc
                )
            )
            power_analysis.append(power_val)

        plt.plot(n, power_analysis, label=f"Effect size = {d}")

    plt.xlabel("Sample Size (per group)")
    plt.ylabel("Statistical Power")
    plt.title("Power Analysis")
    plt.legend()
    plt.grid(True)

    save_figure("power_analysis.png")


def create_hypothesis_testing_flowchart():
    """Create a flowchart for hypothesis testing"""
    plt.figure(figsize=(12, 8))

    # Create graph
    G = nx.DiGraph()

    # Add nodes
    nodes = [
        "Research Question",
        "Formulate Hypotheses",
        "Choose Test",
        "Collect Data",
        "Analyze Results",
        "Draw Conclusion",
    ]

    for node in nodes:
        G.add_node(node)

    # Add edges
    edges = [
        ("Research Question", "Formulate Hypotheses"),
        ("Formulate Hypotheses", "Choose Test"),
        ("Choose Test", "Collect Data"),
        ("Collect Data", "Analyze Results"),
        ("Analyze Results", "Draw Conclusion"),
    ]

    for edge in edges:
        G.add_edge(*edge)

    # Draw graph
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=3000,
        node_color="lightpink",
        font_size=12,
        font_weight="bold",
        arrowsize=20,
    )

    plt.title("Hypothesis Testing Flowchart")
    save_figure("hypothesis_testing_flowchart.png")


def create_null_vs_alternative_distribution():
    """Create visualization of null vs alternative distributions"""
    plt.figure(figsize=(10, 6))

    # Generate data
    np.random.seed(42)
    null_data = np.random.normal(0, 1, 1000)
    alt_data = np.random.normal(1, 1, 1000)

    # Plot distributions
    sns.kdeplot(null_data, label="Null Distribution", shade=True)
    sns.kdeplot(alt_data, label="Alternative Distribution", shade=True)

    # Add critical region
    critical_value = stats.norm.ppf(0.95)
    x = np.linspace(-4, 4, 1000)
    plt.fill_between(
        x[x > critical_value],
        stats.norm.pdf(x[x > critical_value]),
        color="red",
        alpha=0.2,
        label="Critical Region",
    )

    plt.title("Null vs Alternative Distributions")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_figure("null_vs_alternative.png")


def create_multiple_testing_correction():
    """Create visualization of multiple testing correction"""
    plt.figure(figsize=(10, 6))

    # Generate data
    np.random.seed(42)
    p_values = np.random.uniform(0, 1, 20)
    corrected_p_values = p_values * 20  # Bonferroni correction

    # Plot
    plt.scatter(range(len(p_values)), p_values, label="Original p-values")
    plt.scatter(
        range(len(corrected_p_values)), corrected_p_values, label="Corrected p-values"
    )
    plt.axhline(y=0.05, color="r", linestyle="--", label="Significance Level")

    plt.title("Multiple Testing Correction")
    plt.xlabel("Test Number")
    plt.ylabel("p-value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_figure("multiple_testing.png")


def create_results_dashboard():
    """Create a sample results dashboard"""
    plt.figure(figsize=(15, 10))

    # Generate data
    np.random.seed(42)
    days = np.arange(30)
    control = np.random.normal(100, 10, 30)
    treatment = np.random.normal(110, 10, 30)

    # Create subplots
    plt.subplot(221)
    plt.plot(days, control, label="Control")
    plt.plot(days, treatment, label="Treatment")
    plt.title("Daily Performance")
    plt.legend()

    plt.subplot(222)
    sns.boxplot(data=[control, treatment])
    plt.xticks([0, 1], ["Control", "Treatment"])
    plt.title("Distribution Comparison")

    plt.subplot(223)
    effect_size = (np.mean(treatment) - np.mean(control)) / np.std(control)
    plt.bar(["Effect Size"], [effect_size])
    plt.axhline(y=0, color="r", linestyle="--")
    plt.title("Effect Size")

    plt.subplot(224)
    plt.pie(
        [len(control), len(treatment)],
        labels=["Control", "Treatment"],
        autopct="%1.1f%%",
    )
    plt.title("Sample Size Distribution")

    plt.tight_layout()
    save_figure("results_dashboard.png")


def create_decision_framework():
    """Create visualization of decision framework"""
    plt.figure(figsize=(12, 8))

    # Create graph
    G = nx.DiGraph()

    # Add nodes
    nodes = [
        "Statistical Significance",
        "Practical Significance",
        "Implementation Cost",
        "Potential Benefit",
        "Decision",
    ]

    for node in nodes:
        G.add_node(node)

    # Add edges
    edges = [
        ("Statistical Significance", "Decision"),
        ("Practical Significance", "Decision"),
        ("Implementation Cost", "Decision"),
        ("Potential Benefit", "Decision"),
    ]

    for edge in edges:
        G.add_edge(*edge)

    # Draw graph
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=3000,
        node_color="lightyellow",
        font_size=12,
        font_weight="bold",
        arrowsize=20,
    )

    plt.title("Decision Framework")
    save_figure("decision_framework.png")


def create_experimental_design_flowchart():
    """Create a flowchart for experimental design"""
    plt.figure(figsize=(12, 8))

    # Create graph
    G = nx.DiGraph()

    # Add nodes
    nodes = [
        "Research Question",
        "Define Variables",
        "Choose Design",
        "Randomization",
        "Data Collection",
        "Analysis",
    ]

    for node in nodes:
        G.add_node(node)

    # Add edges
    edges = [
        ("Research Question", "Define Variables"),
        ("Define Variables", "Choose Design"),
        ("Choose Design", "Randomization"),
        ("Randomization", "Data Collection"),
        ("Data Collection", "Analysis"),
    ]

    for edge in edges:
        G.add_edge(*edge)

    # Draw graph
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=3000,
        node_color="lightgreen",
        font_size=12,
        font_weight="bold",
        arrowsize=20,
    )

    plt.title("Experimental Design Flowchart")
    save_figure("experimental_design_flowchart.png")


def create_sample_size_visualization():
    """Create visualization for sample size determination"""
    plt.figure(figsize=(10, 6))

    # Generate data
    effect_sizes = np.linspace(0.1, 1.0, 10)
    power_levels = [0.8, 0.9, 0.95]
    alpha = 0.05

    for power in power_levels:
        sample_sizes = []
        for d in effect_sizes:
            # Calculate required sample size
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            z_beta = stats.norm.ppf(power)
            n = 2 * ((z_alpha + z_beta) / d) ** 2
            sample_sizes.append(n)

        plt.plot(effect_sizes, sample_sizes, label=f"Power = {power}")

    plt.xlabel("Effect Size (d)")
    plt.ylabel("Required Sample Size (per group)")
    plt.title("Sample Size Determination")
    plt.legend()
    plt.grid(True)

    save_figure("sample_size_determination.png")


if __name__ == "__main__":
    # Create all visualizations
    create_scientific_method_flowchart()
    create_statistical_test_decision_tree()
    create_ab_testing_timeline()
    create_effect_size_visualization()
    create_confidence_interval_diagram()
    create_power_analysis_visualization()
    create_hypothesis_testing_flowchart()
    create_null_vs_alternative_distribution()
    create_multiple_testing_correction()
    create_results_dashboard()
    create_decision_framework()
    create_experimental_design_flowchart()
    create_sample_size_visualization()
