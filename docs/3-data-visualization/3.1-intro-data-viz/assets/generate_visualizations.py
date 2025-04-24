from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Set style for better-looking plots
plt.style.use("default")  # Use default style instead of seaborn


def create_chart_selection_guide():
    """Create a visual guide for chart selection"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a grid of examples
    categories = ["Comparison", "Distribution", "Relationship", "Composition"]
    examples = [
        ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart"],
        ["Histogram", "Box Plot", "Bubble Chart", "Stacked Area"],
        ["Lollipop Chart", "Violin Plot", "Heatmap", "Treemap"],
    ]

    # Plot each example
    for i, category in enumerate(categories):
        for j, example in enumerate(examples[i % len(examples)]):
            ax.text(
                i,
                j,
                f"{category}\n{example}",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.8),
            )

    ax.set_title("Chart Selection Guide", pad=20)
    ax.axis("off")

    # Save the figure
    plt.savefig("chart_selection_guide.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_visual_hierarchy_example():
    """Create an example of visual hierarchy"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # Plot main data (primary)
    ax.plot(x, y1, "b-", linewidth=2, label="Primary Data")

    # Plot secondary data
    ax.plot(x, y2, "r--", linewidth=1, label="Secondary Data")

    # Add annotations
    ax.annotate(
        "Primary Message",
        xy=(5, 0),
        xytext=(5, 1),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )

    ax.set_title("Visual Hierarchy Example", pad=20)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.7)

    # Save the figure
    plt.savefig("visual_hierarchy.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_color_scheme_examples():
    """Create examples of different color schemes"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Sequential color scheme
    data1 = np.random.rand(10, 10)
    im1 = ax1.imshow(data1, cmap="Blues")
    ax1.set_title("Sequential\n(Light to Dark)")
    plt.colorbar(im1, ax=ax1)

    # Diverging color scheme
    data2 = np.random.randn(10, 10)
    im2 = ax2.imshow(data2, cmap="RdBu")
    ax2.set_title("Diverging\n(Red to Blue)")
    plt.colorbar(im2, ax=ax2)

    # Qualitative color scheme
    data3 = np.random.randint(0, 5, (10, 10))
    im3 = ax3.imshow(data3, cmap="Set3")
    ax3.set_title("Qualitative\n(Distinct Colors)")
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout()

    # Save the figure
    plt.savefig("color_schemes.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_before_after_example():
    """Create a before/after visualization example"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Generate sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.2, 100)

    # Before: Basic plot
    ax1.plot(x, y, "b-")
    ax1.set_title("Before: Basic Plot")
    ax1.grid(True)

    # After: Enhanced plot
    ax2.plot(x, y, "b-", alpha=0.5)
    ax2.plot(x, np.sin(x), "r--", label="Trend")
    ax2.fill_between(x, y, np.sin(x), alpha=0.2)
    ax2.set_title("After: Enhanced Plot")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()

    # Save the figure
    plt.savefig("before_after.png", bbox_inches="tight", dpi=300)
    plt.close()


def main():
    """Generate all visualization examples"""
    # Create assets directory if it doesn't exist
    Path(".").mkdir(exist_ok=True)

    # Generate all examples
    create_chart_selection_guide()
    create_visual_hierarchy_example()
    create_color_scheme_examples()
    create_before_after_example()


if __name__ == "__main__":
    main()
    main()
