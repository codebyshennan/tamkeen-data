from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for better readability
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def create_story_arc_visualization():
    """Create a visualization of the story arc concept"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create data points for the story arc
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + 1.5

    # Plot the story arc
    ax.plot(x, y, "b-", linewidth=2)

    # Add labels for key points
    points = [
        (0, "Setup"),
        (2.5, "Rising Action"),
        (5, "Climax"),
        (7.5, "Falling Action"),
        (10, "Resolution"),
    ]
    for x_pos, label in points:
        ax.text(
            x_pos,
            1.2,
            label,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7),
        )

    # Add title and labels
    ax.set_title("The Story Arc in Data Storytelling", pad=20)
    ax.set_xlabel("Story Progress")
    ax.set_ylabel("Engagement Level")

    # Remove unnecessary elements
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the figure
    plt.savefig("story_arc.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_visual_hierarchy_diagram():
    """Create a visualization showing visual hierarchy in data storytelling"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create sample data
    categories = ["Primary", "Secondary", "Tertiary"]
    values = [100, 60, 30]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    # Create horizontal bar chart
    bars = ax.barh(categories, values, color=colors)

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 5, bar.get_y() + bar.get_height() / 2, f"{width}%", va="center")

    # Add title and labels
    ax.set_title("Visual Hierarchy in Data Storytelling", pad=20)
    ax.set_xlabel("Importance Level")

    # Remove unnecessary elements
    ax.set_xticks([])

    # Save the figure
    plt.savefig("visual_hierarchy.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_chart_selection_guide():
    """Create a comprehensive chart selection guide with examples"""
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 20))

    # Create a grid of subplots
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

    # 1. Comparison Charts
    ax1 = fig.add_subplot(gs[0, 0])
    create_bar_chart_example(ax1)
    ax1.set_title("Comparison Charts\n(Bar, Column, Bullet)", pad=20)

    # 2. Trend Charts
    ax2 = fig.add_subplot(gs[0, 1])
    create_line_chart_example(ax2)
    ax2.set_title("Trend Charts\n(Line, Area, Sparkline)", pad=20)

    # 3. Distribution Charts
    ax3 = fig.add_subplot(gs[1, 0])
    create_histogram_example(ax3)
    ax3.set_title("Distribution Charts\n(Histogram, Box Plot, Violin)", pad=20)

    # 4. Relationship Charts
    ax4 = fig.add_subplot(gs[1, 1])
    create_scatter_plot_example(ax4)
    ax4.set_title("Relationship Charts\n(Scatter, Bubble, Heat Map)", pad=20)

    # 5. Composition Charts
    ax5 = fig.add_subplot(gs[2, 0])
    create_pie_chart_example(ax5)
    ax5.set_title("Composition Charts\n(Pie, Donut, Stacked Bar)", pad=20)

    # 6. Hierarchical Charts
    ax6 = fig.add_subplot(gs[2, 1])
    create_treemap_example(ax6)
    ax6.set_title("Hierarchical Charts\n(Treemap, Sunburst, Tree)", pad=20)

    # 7. Decision Tree
    ax7 = fig.add_subplot(gs[3, :])
    create_decision_tree(ax7)
    ax7.set_title("Chart Selection Decision Tree", pad=20)

    # Save the figure
    plt.savefig("chart_selection.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_bar_chart_example(ax):
    """Create a bar chart example"""
    categories = ["Product A", "Product B", "Product C", "Product D"]
    values = [45, 30, 25, 15]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

    bars = ax.bar(categories, values, color=colors)
    ax.set_ylim(0, 50)
    ax.set_ylabel("Sales (units)")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height}",
            ha="center",
            va="bottom",
        )


def create_line_chart_example(ax):
    """Create a line chart example"""
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x) + 2
    y2 = np.cos(x) + 2

    ax.plot(x, y1, "b-", label="Product A")
    ax.plot(x, y2, "r--", label="Product B")
    ax.set_ylim(0, 4)
    ax.legend()
    ax.set_ylabel("Value")


def create_histogram_example(ax):
    """Create a histogram example"""
    data = np.random.normal(0, 1, 1000)
    ax.hist(data, bins=30, color="#4ECDC4", alpha=0.7)
    ax.set_ylabel("Frequency")


def create_scatter_plot_example(ax):
    """Create a scatter plot example"""
    x = np.random.normal(0, 1, 100)
    y = x + np.random.normal(0, 0.5, 100)
    size = np.random.uniform(10, 100, 100)

    scatter = ax.scatter(x, y, s=size, alpha=0.6, c="#45B7D1")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def create_pie_chart_example(ax):
    """Create a pie chart example"""
    sizes = [30, 25, 20, 15, 10]
    labels = ["A", "B", "C", "D", "E"]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEEAD"]

    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%")
    ax.axis("equal")


def create_treemap_example(ax):
    """Create a simple treemap-like visualization"""
    rectangles = [
        {"x": 0, "y": 0, "width": 0.6, "height": 0.6, "color": "#FF6B6B"},
        {"x": 0.6, "y": 0, "width": 0.4, "height": 0.4, "color": "#4ECDC4"},
        {"x": 0.6, "y": 0.4, "width": 0.4, "height": 0.2, "color": "#45B7D1"},
        {"x": 0, "y": 0.6, "width": 0.3, "height": 0.4, "color": "#96CEB4"},
        {"x": 0.3, "y": 0.6, "width": 0.7, "height": 0.4, "color": "#FFEEAD"},
    ]

    for rect in rectangles:
        ax.add_patch(
            plt.Rectangle(
                (rect["x"], rect["y"]),
                rect["width"],
                rect["height"],
                color=rect["color"],
            )
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def create_decision_tree(ax):
    """Create a decision tree for chart selection"""
    # Create a simple decision tree
    tree = {
        "What do you want to show?": {
            "Comparison": {
                "Few Categories": "Bar Chart",
                "Many Categories": "Horizontal Bar Chart",
            },
            "Trends": {"Over Time": "Line Chart", "Distribution": "Histogram"},
            "Relationships": {
                "Two Variables": "Scatter Plot",
                "Three Variables": "Bubble Chart",
            },
            "Composition": {"Parts of Whole": "Pie Chart", "Hierarchy": "Treemap"},
        }
    }

    # Create a simple visualization of the decision tree
    y_pos = 0
    for main_cat, subcats in tree["What do you want to show?"].items():
        ax.text(0.1, y_pos, main_cat, fontsize=12, weight="bold")
        for subcat, chart in subcats.items():
            ax.text(0.3, y_pos - 0.1, f"→ {subcat}: {chart}", fontsize=10)
            y_pos -= 0.2
        y_pos -= 0.3

    # Remove unnecessary elements
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")


def main():
    """Generate all visualizations"""
    # Create output directory if it doesn't exist
    output_dir = Path("docs/3-data-visualization/3.4-data-storytelling/assets")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    create_story_arc_visualization()
    create_visual_hierarchy_diagram()
    create_chart_selection_guide()


if __name__ == "__main__":
    main()
