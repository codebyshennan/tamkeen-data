import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle, Rectangle


def create_story_structure_diagram():
    """Create a visualization of different story structure frameworks"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Classic Narrative Arc
    x = np.linspace(0, 10, 100)
    y = 2 * np.exp(-((x - 5) ** 2) / 8)

    # Plot the arc with gradient
    gradient = np.linspace(0, 1, len(x))
    for i in range(len(x) - 1):
        ax1.plot(
            x[i : i + 2], y[i : i + 2], color=plt.cm.Blues(gradient[i]), linewidth=3
        )

    # Add labels
    points = [
        (0.5, 0.2, "Hook", "Grab attention"),
        (2.5, 1.2, "Setup", "Establish context"),
        (5.0, 2.0, "Journey", "Build tension"),
        (7.5, 1.2, "Reveal", "Share insights"),
        (9.5, 0.2, "Call to Action", "Drive action"),
    ]

    for x, y, label, desc in points:
        ax1.plot(x, y, "o", color="#FF6B6B", markersize=10)
        ax1.annotate(
            label,
            (x, y),
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            fontsize=10,
        )
        ax1.annotate(
            desc,
            (x, y),
            xytext=(0, -25),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="gray",
        )

    ax1.set_title("Classic Narrative Arc", pad=20, fontsize=14, fontweight="bold")
    ax1.set_xlim(-0.5, 10.5)
    ax1.set_ylim(-0.5, 2.5)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.grid(True, linestyle="--", alpha=0.3)

    # SCR Framework
    stages = ["Situation", "Complication", "Resolution"]
    values = [1, 2, 3]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    bars = ax2.barh(stages, values, color=colors, height=0.6)

    # Add descriptions
    descriptions = [
        "Current state and context",
        "Challenge or problem to solve",
        "Solution and next steps",
    ]

    for i, (bar, desc) in enumerate(zip(bars, descriptions)):
        ax2.text(
            1.5,
            bar.get_y() + bar.get_height() / 2,
            desc,
            va="center",
            ha="left",
            fontsize=10,
            color="gray",
        )

    ax2.set_title("SCR Framework", pad=20, fontsize=14, fontweight="bold")
    ax2.set_xlim(0, 4)
    ax2.set_xticks([])
    ax2.grid(True, linestyle="--", alpha=0.3)

    # Add a light background color to all subplots
    for ax in [ax1, ax2]:
        ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")

    plt.tight_layout()
    plt.savefig("story_structure.png", bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()


def create_visualization_decision_tree():
    """Create a decision tree for choosing the right visualization type"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define nodes and their positions
    nodes = [
        (0.5, 0.9, "What's your goal?", "Start here"),
        (0.2, 0.7, "Compare values", "Show differences"),
        (0.5, 0.7, "Show trends", "Time series"),
        (0.8, 0.7, "Show distribution", "Data spread"),
        (0.1, 0.5, "Bar Chart", "Categories"),
        (0.3, 0.5, "Line Chart", "Time series"),
        (0.7, 0.5, "Histogram", "Distribution"),
        (0.9, 0.5, "Box Plot", "Outliers"),
    ]

    # Draw nodes
    for x, y, label, desc in nodes:
        ax.add_patch(Circle((x, y), 0.05, color="#FF6B6B", alpha=0.3))
        ax.text(x, y, label, ha="center", va="center", fontweight="bold")
        ax.text(x, y - 0.05, desc, ha="center", va="center", fontsize=8, color="gray")

    # Draw connections
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

    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1)
    ax.set_title("Visualization Decision Tree", pad=20, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a light background color
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")

    plt.savefig(
        "visualization_decision_tree.png",
        bbox_inches="tight",
        dpi=300,
        facecolor="white",
    )
    plt.close()


def create_story_creation_process():
    """Create a flowchart of the data story creation process"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define process steps
    steps = [
        (0.1, 0.8, "Data Collection", "Gather relevant data"),
        (0.3, 0.8, "Analysis", "Identify patterns"),
        (0.5, 0.8, "Insight Generation", "Find key insights"),
        (0.7, 0.8, "Story Structure", "Organize narrative"),
        (0.9, 0.8, "Visualization", "Create visuals"),
        (0.9, 0.6, "Review", "Get feedback"),
        (0.7, 0.6, "Refine", "Improve story"),
        (0.5, 0.6, "Finalize", "Complete story"),
        (0.3, 0.6, "Present", "Share insights"),
        (0.1, 0.6, "Measure Impact", "Track results"),
    ]

    # Draw process boxes
    for x, y, label, desc in steps:
        ax.add_patch(
            Rectangle((x - 0.1, y - 0.1), 0.2, 0.2, color="#FF6B6B", alpha=0.3)
        )
        ax.text(x, y, label, ha="center", va="center", fontweight="bold")
        ax.text(x, y - 0.05, desc, ha="center", va="center", fontsize=8, color="gray")

    # Draw connections
    for i in range(len(steps) - 1):
        x1, y1 = steps[i][0], steps[i][1]
        x2, y2 = steps[i + 1][0], steps[i + 1][1]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5),
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 1)
    ax.set_title("Data Story Creation Process", pad=20, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a light background color
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")

    plt.savefig(
        "story_creation_process.png", bbox_inches="tight", dpi=300, facecolor="white"
    )
    plt.close()


def create_color_palette_guide():
    """Create a visualization of recommended color palettes"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))

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

    # Accessible
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEEAD"]
    for i, color in enumerate(colors):
        axes[3].add_patch(Rectangle((i, 0), 1, 1, color=color))
    axes[3].set_title("Accessible Color Scheme", pad=20, fontsize=12, fontweight="bold")
    axes[3].text(
        2, 1.2, "Use for color-blind friendly visuals", ha="center", fontsize=10
    )
    axes[3].set_xticks([])
    axes[3].set_yticks([])

    # Add a light background color to all subplots
    for ax in axes:
        ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(
        "color_palette_guide.png", bbox_inches="tight", dpi=300, facecolor="white"
    )
    plt.close()


def create_quality_checklist():
    """Create a visualization of the story quality checklist"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define checklist items
    items = [
        "Clear main message",
        "Logical flow",
        "Appropriate visuals",
        "Consistent styling",
        "Accessible design",
        "Actionable insights",
        "Engaging narrative",
        "Proper context",
        "Clear call to action",
        "Impact measurement",
    ]

    # Create checklist boxes
    for i, item in enumerate(items):
        y = 0.9 - (i * 0.08)
        ax.add_patch(Rectangle((0.1, y - 0.03), 0.05, 0.05, color="#FF6B6B", alpha=0.3))
        ax.text(0.2, y, item, ha="left", va="center", fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Story Quality Checklist", pad=20, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a light background color
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")

    plt.savefig(
        "quality_checklist.png", bbox_inches="tight", dpi=300, facecolor="white"
    )
    plt.close()


def main():
    """Generate all visual enhancements"""
    create_story_structure_diagram()
    create_visualization_decision_tree()
    create_story_creation_process()
    create_color_palette_guide()
    create_quality_checklist()


if __name__ == "__main__":
    main()
