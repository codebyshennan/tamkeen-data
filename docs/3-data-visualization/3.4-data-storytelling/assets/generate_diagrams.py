import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle, Rectangle


def create_narrative_arc():
    """Create a visualization of the narrative arc in data storytelling"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a more natural narrative arc using a custom curve
    x = np.linspace(0, 10, 100)
    y = 2 * np.exp(-((x - 5) ** 2) / 8)  # Gaussian-like curve for a natural arc

    # Plot the arc with a gradient color
    gradient = np.linspace(0, 1, len(x))
    for i in range(len(x) - 1):
        ax.plot(
            x[i : i + 2], y[i : i + 2], color=plt.cm.Blues(gradient[i]), linewidth=3
        )

    # Define key points in the narrative arc
    points = [
        (0.5, 0.2, "Hook", "Grab attention"),
        (2.5, 1.2, "Setup", "Establish context"),
        (5.0, 2.0, "Journey", "Build tension"),
        (7.5, 1.2, "Reveal", "Share insights"),
        (9.5, 0.2, "Call to Action", "Drive action"),
    ]

    # Add points and annotations
    for x, y, label, desc in points:
        ax.plot(x, y, "o", color="#FF6B6B", markersize=10)
        ax.annotate(
            label,
            (x, y),
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            fontsize=10,
        )
        ax.annotate(
            desc,
            (x, y),
            xytext=(0, -25),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="gray",
        )

    # Customize the plot
    ax.set_title(
        "Narrative Arc in Data Storytelling", pad=20, fontsize=14, fontweight="bold"
    )
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 2.5)

    # Remove axes and add a subtle grid
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, linestyle="--", alpha=0.3)

    # Add a light background color
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")

    plt.savefig("narrative_arc.png", bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()


def create_visual_hierarchy():
    """Create a visualization showing visual hierarchy in data storytelling"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create sample data with more descriptive categories
    categories = ["Primary Message", "Supporting Data", "Additional Context"]
    values = [100, 65, 35]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    # Create horizontal bar chart with enhanced styling
    bars = ax.barh(categories, values, color=colors, height=0.6)

    # Add value labels with percentage signs
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 2,
            bar.get_y() + bar.get_height() / 2,
            f"{width}%",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Add descriptive text for each level
    descriptions = [
        "Main insight or key finding",
        "Supporting evidence and data points",
        "Background information and context",
    ]

    for i, (bar, desc) in enumerate(zip(bars, descriptions)):
        ax.text(
            5,
            bar.get_y() + bar.get_height() / 2,
            desc,
            va="center",
            ha="left",
            fontsize=9,
            color="gray",
        )

    # Customize the plot
    ax.set_title(
        "Visual Hierarchy in Data Storytelling", pad=20, fontsize=14, fontweight="bold"
    )
    ax.set_xlim(0, 110)
    ax.set_xticks([])
    ax.grid(True, linestyle="--", alpha=0.3)

    # Add a light background color
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")

    plt.savefig("visual_hierarchy.png", bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()


def create_color_schemes():
    """Create a visualization of different color schemes"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Sequential
    colors = sns.color_palette("Blues", n_colors=5)
    for i, color in enumerate(colors):
        axes[0].add_patch(Rectangle((i, 0), 1, 1, color=color))
    axes[0].set_title("Sequential Color Scheme", pad=20, fontsize=12, fontweight="bold")
    axes[0].text(2, 1.2, "Use for ordered data", ha="center", fontsize=10)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Categorical
    colors = sns.color_palette("Set2", n_colors=5)
    for i, color in enumerate(colors):
        axes[1].add_patch(Rectangle((i, 0), 1, 1, color=color))
    axes[1].set_title(
        "Categorical Color Scheme", pad=20, fontsize=12, fontweight="bold"
    )
    axes[1].text(2, 1.2, "Use for distinct categories", ha="center", fontsize=10)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Diverging
    colors = sns.color_palette("RdBu", n_colors=5)
    for i, color in enumerate(colors):
        axes[2].add_patch(Rectangle((i, 0), 1, 1, color=color))
    axes[2].set_title("Diverging Color Scheme", pad=20, fontsize=12, fontweight="bold")
    axes[2].text(2, 1.2, "Use for positive/negative values", ha="center", fontsize=10)
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    # Add a light background color to all subplots
    for ax in axes:
        ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")

    plt.tight_layout()
    plt.savefig("color_schemes.png", bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()


def create_layout_examples():
    """Create visualizations of different layout examples"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Dashboard Layout
    axes[0].add_patch(Rectangle((0, 0), 1, 0.3, color="#FF6B6B", alpha=0.3))
    axes[0].add_patch(Rectangle((0, 0.3), 1, 0.4, color="#4ECDC4", alpha=0.3))
    axes[0].add_patch(Rectangle((0, 0.7), 1, 0.3, color="#45B7D1", alpha=0.3))
    axes[0].text(0.5, 0.15, "Key Metrics", ha="center", fontweight="bold")
    axes[0].text(0.5, 0.5, "Detailed Analysis", ha="center", fontweight="bold")
    axes[0].text(0.5, 0.85, "Additional Details", ha="center", fontweight="bold")
    axes[0].set_title("Dashboard Layout", pad=20, fontsize=12, fontweight="bold")
    axes[0].text(
        0.5,
        -0.1,
        "Ideal for real-time monitoring and quick insights",
        ha="center",
        fontsize=10,
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Report Layout
    axes[1].add_patch(Rectangle((0, 0), 1, 0.2, color="#FF6B6B", alpha=0.3))
    axes[1].add_patch(Rectangle((0, 0.2), 1, 0.6, color="#4ECDC4", alpha=0.3))
    axes[1].add_patch(Rectangle((0, 0.8), 1, 0.2, color="#45B7D1", alpha=0.3))
    axes[1].text(0.5, 0.1, "Header", ha="center", fontweight="bold")
    axes[1].text(0.5, 0.5, "Body", ha="center", fontweight="bold")
    axes[1].text(0.5, 0.9, "Footer", ha="center", fontweight="bold")
    axes[1].set_title("Report Layout", pad=20, fontsize=12, fontweight="bold")
    axes[1].text(
        0.5,
        -0.1,
        "Perfect for detailed analysis and documentation",
        ha="center",
        fontsize=10,
    )
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Presentation Layout
    axes[2].add_patch(Rectangle((0, 0), 1, 0.2, color="#FF6B6B", alpha=0.3))
    axes[2].add_patch(Rectangle((0, 0.2), 1, 0.6, color="#4ECDC4", alpha=0.3))
    axes[2].add_patch(Rectangle((0, 0.8), 1, 0.2, color="#45B7D1", alpha=0.3))
    axes[2].text(0.5, 0.1, "Opening", ha="center", fontweight="bold")
    axes[2].text(0.5, 0.5, "Middle", ha="center", fontweight="bold")
    axes[2].text(0.5, 0.9, "Closing", ha="center", fontweight="bold")
    axes[2].set_title("Presentation Layout", pad=20, fontsize=12, fontweight="bold")
    axes[2].text(
        0.5,
        -0.1,
        "Best for engaging storytelling and audience interaction",
        ha="center",
        fontsize=10,
    )
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    # Add a light background color to all subplots
    for ax in axes:
        ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")

    plt.tight_layout()
    plt.savefig("layout_examples.png", bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()


def create_chart_selection():
    """Create a visualization of chart selection guide"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a decision tree with enhanced styling
    nodes = [
        (0.5, 0.9, "What's your goal?", "Start here"),
        (0.2, 0.7, "Comparison", "Compare values"),
        (0.5, 0.7, "Trend", "Show changes over time"),
        (0.8, 0.7, "Distribution", "Show data spread"),
        (0.1, 0.5, "Bar Chart", "Compare categories"),
        (0.3, 0.5, "Line Chart", "Show trends"),
        (0.7, 0.5, "Histogram", "Show distribution"),
        (0.9, 0.5, "Box Plot", "Show outliers"),
    ]

    # Draw nodes with enhanced styling
    for x, y, label, desc in nodes:
        ax.add_patch(Circle((x, y), 0.05, color="#FF6B6B", alpha=0.3))
        ax.text(x, y, label, ha="center", va="center", fontweight="bold")
        ax.text(x, y - 0.05, desc, ha="center", va="center", fontsize=8, color="gray")

    # Draw connections with arrows
    connections = [
        (0.5, 0.9, 0.2, 0.7),
        (0.5, 0.9, 0.5, 0.7),
        (0.5, 0.9, 0.8, 0.7),
        (0.2, 0.7, 0.1, 0.5),
        (0.5, 0.7, 0.3, 0.5),
        (0.8, 0.7, 0.7, 0.5),
        (0.8, 0.7, 0.9, 0.5),
    ]

    for x1, y1, x2, y2 in connections:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5),
        )

    # Customize the plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1)
    ax.set_title("Chart Selection Guide", pad=20, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a light background color
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")

    plt.savefig("chart_selection.png", bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()


def main():
    """Generate all diagrams"""
    create_narrative_arc()
    create_visual_hierarchy()
    create_color_schemes()
    create_layout_examples()
    create_chart_selection()


if __name__ == "__main__":
    main()
