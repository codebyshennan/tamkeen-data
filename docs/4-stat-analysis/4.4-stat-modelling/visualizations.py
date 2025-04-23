import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

# Create assets directory if it doesn't exist
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def set_style():
    """Set consistent style for all plots"""
    # Use a valid matplotlib style
    plt.style.use("default")

    # Set custom style parameters
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["figure.titlesize"] = 18

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Set grid style
    plt.rcParams["grid.color"] = "0.8"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.5

    # Set figure background
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    # Set spines
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def save_figure(filename):
    """Save figure to assets directory with high DPI"""
    plt.savefig(os.path.join(ASSETS_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


def create_model_selection_flowchart():
    """Create a flowchart for model selection process"""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig)

    # Create boxes with descriptions
    boxes = {
        "data": {"title": "Data Collection", "desc": "Gather and understand your data"},
        "preprocess": {"title": "Preprocessing", "desc": "Clean and prepare data"},
        "model": {"title": "Model Building", "desc": "Select and train models"},
        "evaluate": {"title": "Evaluation", "desc": "Assess model performance"},
        "deploy": {"title": "Deployment", "desc": "Implement and monitor"},
    }

    # Draw boxes
    for name, info in boxes.items():
        ax = plt.subplot(
            gs[
                (
                    0
                    if name == "data"
                    else 1 if name in ["preprocess", "model", "evaluate"] else 2
                ),
                (
                    1
                    if name == "data"
                    else (
                        0
                        if name == "preprocess"
                        else 1 if name == "model" else 2 if name == "evaluate" else 1
                    )
                ),
            ]
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.8, fc="lightblue", ec="black"))
        ax.text(
            0.5,
            0.6,
            info["title"],
            ha="center",
            va="center",
            fontsize=14,
            weight="bold",
        )
        ax.text(
            0.5,
            0.4,
            info["desc"],
            ha="center",
            va="center",
            fontsize=10,
            style="italic",
        )

    # Add arrows with labels
    arrow_props = dict(arrowstyle="->", color="black", lw=2)

    # Data to Preprocess
    arrow = FancyArrowPatch(
        (0.5, 0.1), (0.5, 0.9), connectionstyle="arc3,rad=0.2", **arrow_props
    )
    plt.gca().add_patch(arrow)
    plt.text(0.5, 0.5, "Prepare", ha="center", va="center", fontsize=10)

    # Preprocess to Model
    arrow = FancyArrowPatch(
        (0.9, 0.5), (0.1, 0.5), connectionstyle="arc3,rad=0.2", **arrow_props
    )
    plt.gca().add_patch(arrow)
    plt.text(0.5, 0.5, "Transform", ha="center", va="center", fontsize=10)

    # Model to Evaluate
    arrow = FancyArrowPatch(
        (0.9, 0.5), (0.1, 0.5), connectionstyle="arc3,rad=0.2", **arrow_props
    )
    plt.gca().add_patch(arrow)
    plt.text(0.5, 0.5, "Test", ha="center", va="center", fontsize=10)

    # Evaluate to Deploy
    arrow = FancyArrowPatch(
        (0.5, 0.1), (0.5, 0.9), connectionstyle="arc3,rad=0.2", **arrow_props
    )
    plt.gca().add_patch(arrow)
    plt.text(0.5, 0.5, "Implement", ha="center", va="center", fontsize=10)

    plt.suptitle("Model Selection Process Flowchart", fontsize=20)
    plt.tight_layout()
    save_figure("model_selection_flowchart.png")


def create_regularization_comparison():
    """Create comparison infographic for regularization techniques"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # L1 Regularization
    x = np.linspace(-2, 2, 100)
    y = np.abs(x)
    ax1.plot(x, y, "b-", lw=2)
    ax1.set_title("L1 Regularization (Lasso)")
    ax1.set_xlabel("Coefficient Value")
    ax1.set_ylabel("Penalty")
    ax1.grid(True)
    ax1.text(
        0,
        0.5,
        "Feature Selection\n(Sparse Solutions)\n\n• Sets some coefficients to zero\n• Good for feature selection\n• Less stable with correlated features",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # L2 Regularization
    y = x**2
    ax2.plot(x, y, "r-", lw=2)
    ax2.set_title("L2 Regularization (Ridge)")
    ax2.set_xlabel("Coefficient Value")
    ax2.set_ylabel("Penalty")
    ax2.grid(True)
    ax2.text(
        0,
        0.5,
        "Smooth Shrinkage\n(No Feature Selection)\n\n• Shrinks all coefficients\n• More stable with correlated features\n• Better for prediction",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.suptitle("Regularization Techniques Comparison", fontsize=16)
    plt.tight_layout()
    save_figure("regularization_comparison.png")


def create_model_complexity_guide():
    """Create visual guide for model complexity"""
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 3, figure=fig)

    # Generate sample data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.2, 100)

    # Plot different model complexities
    degrees = [1, 3, 10]
    titles = ["Underfitting", "Good Fit", "Overfitting"]
    descriptions = [
        "Too simple to capture patterns\nHigh bias, low variance",
        "Captures main patterns\nBalanced bias and variance",
        "Fits noise in data\nLow bias, high variance",
    ]

    for i, (degree, title, desc) in enumerate(zip(degrees, titles, descriptions)):
        ax = plt.subplot(gs[0, i])

        # Fit polynomial
        p = np.polyfit(x, y, degree)
        y_pred = np.polyval(p, x)

        # Plot
        ax.scatter(x, y, alpha=0.5)
        ax.plot(x, y_pred, "r-", lw=2)
        ax.set_title(title)
        ax.text(
            0.5, -0.2, desc, ha="center", va="top", transform=ax.transAxes, fontsize=10
        )
        ax.grid(True)

    # Add bias-variance tradeoff
    ax = plt.subplot(gs[1, :])
    complexity = np.linspace(0, 10, 100)
    bias = 1 / (complexity + 1)
    variance = complexity / 10
    total_error = bias + variance

    ax.plot(complexity, bias, "b-", label="Bias", lw=2)
    ax.plot(complexity, variance, "r-", label="Variance", lw=2)
    ax.plot(complexity, total_error, "g-", label="Total Error", lw=2)

    ax.set_xlabel("Model Complexity")
    ax.set_ylabel("Error")
    ax.set_title("Bias-Variance Tradeoff")
    ax.legend()
    ax.grid(True)

    # Add optimal point
    optimal = complexity[np.argmin(total_error)]
    ax.axvline(
        x=optimal,
        color="k",
        linestyle="--",
        label=f"Optimal Complexity ({optimal:.1f})",
    )
    ax.legend()

    plt.tight_layout()
    save_figure("model_complexity_guide.png")


def create_logistic_regression_guide():
    """Create visual guide for logistic regression"""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)

    # Sigmoid Function
    ax1 = plt.subplot(gs[0, 0])
    x = np.linspace(-6, 6, 100)
    y = 1 / (1 + np.exp(-x))
    ax1.plot(x, y, "b-", lw=2)
    ax1.set_title("Sigmoid Function")
    ax1.set_xlabel("Linear Combination of Features")
    ax1.set_ylabel("Probability")
    ax1.grid(True)

    # Add annotations
    ax1.annotate(
        "Almost Certain 0",
        xy=(-4, 0.02),
        xytext=(-5, 0.1),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )
    ax1.annotate(
        "Decision Boundary",
        xy=(0, 0.5),
        xytext=(1, 0.6),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )
    ax1.annotate(
        "Almost Certain 1",
        xy=(4, 0.98),
        xytext=(3, 0.9),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )

    # Decision Boundary
    ax2 = plt.subplot(gs[0, 1])
    x1 = np.random.normal(0, 1, 100)
    x2 = np.random.normal(0, 1, 100)
    y = (x1 + x2 > 0).astype(int)

    scatter = ax2.scatter(x1, x2, c=y, cmap="coolwarm", alpha=0.6)
    ax2.set_title("Decision Boundary")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")
    ax2.grid(True)

    # Add decision boundary line
    x_line = np.linspace(-3, 3, 100)
    y_line = -x_line
    ax2.plot(x_line, y_line, "k--", label="Decision Boundary")
    ax2.legend()

    # Odds Ratio
    ax3 = plt.subplot(gs[1, 0])
    coef = np.linspace(-2, 2, 100)
    odds = np.exp(coef)
    ax3.plot(coef, odds, "g-", lw=2)
    ax3.set_title("Odds Ratio")
    ax3.set_xlabel("Coefficient")
    ax3.set_ylabel("Odds Ratio")
    ax3.grid(True)

    # Add annotations
    ax3.annotate(
        "Negative Effect",
        xy=(-1, np.exp(-1)),
        xytext=(-1.5, 0.5),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )
    ax3.annotate(
        "Positive Effect",
        xy=(1, np.exp(1)),
        xytext=(1.5, 2),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )

    # Confusion Matrix
    ax4 = plt.subplot(gs[1, 1])
    cm = np.array([[45, 5], [10, 40]])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax4)
    ax4.set_title("Confusion Matrix")
    ax4.set_xlabel("Predicted")
    ax4.set_ylabel("Actual")

    # Add metrics
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    ax4.text(
        0.5,
        -0.2,
        f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}",
        ha="center",
        va="top",
        transform=ax4.transAxes,
        fontsize=10,
    )

    plt.tight_layout()
    save_figure("logistic_regression_guide.png")


def create_polynomial_regression_guide():
    """Create visual guide for polynomial regression"""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)

    # Generate sample data
    np.random.seed(42)
    x = np.linspace(-3, 3, 100)
    y = x**3 - 2 * x**2 + x + np.random.normal(0, 0.5, 100)

    # Plot different degrees
    degrees = [1, 2, 3, 10]
    descriptions = [
        "Linear (Underfitting)\nToo simple for the data",
        "Quadratic\nBetter fit but still simple",
        "Cubic\nGood balance of complexity",
        "High Degree (Overfitting)\nFits noise in the data",
    ]

    for i, (degree, desc) in enumerate(zip(degrees, descriptions)):
        ax = plt.subplot(gs[i // 2, i % 2])

        # Fit polynomial
        p = np.polyfit(x, y, degree)
        y_pred = np.polyval(p, x)

        # Plot
        ax.scatter(x, y, alpha=0.5)
        ax.plot(x, y_pred, "r-", lw=2)
        ax.set_title(f"Degree {degree} Polynomial")
        ax.text(
            0.5, -0.2, desc, ha="center", va="top", transform=ax.transAxes, fontsize=10
        )
        ax.grid(True)

    plt.tight_layout()
    save_figure("polynomial_regression_guide.png")


def create_model_interpretation_guide():
    """Create visual guide for model interpretation"""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)

    # Feature Importance
    ax1 = plt.subplot(gs[0, 0])
    features = ["Age", "Income", "Education", "Experience"]
    importance = [0.8, 0.6, 0.4, 0.2]
    ax1.barh(features, importance)
    ax1.set_title("Feature Importance")
    ax1.set_xlabel("Importance Score")
    ax1.grid(True)
    ax1.text(
        0.5,
        -0.2,
        "Shows which features have the most impact on predictions",
        ha="center",
        va="top",
        transform=ax1.transAxes,
        fontsize=10,
    )

    # Partial Dependence
    ax2 = plt.subplot(gs[0, 1])
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + 0.1 * x
    ax2.plot(x, y, "b-", lw=2)
    ax2.set_title("Partial Dependence Plot")
    ax2.set_xlabel("Feature Value")
    ax2.set_ylabel("Effect on Prediction")
    ax2.grid(True)
    ax2.text(
        0.5,
        -0.2,
        "Shows how a feature affects predictions\nwhile averaging out other features",
        ha="center",
        va="top",
        transform=ax2.transAxes,
        fontsize=10,
    )

    # SHAP Values
    ax3 = plt.subplot(gs[1, 0])
    features = ["Feature 1", "Feature 2", "Feature 3", "Feature 4"]
    shap_values = [0.3, -0.2, 0.1, -0.4]
    ax3.barh(features, shap_values)
    ax3.set_title("SHAP Values")
    ax3.set_xlabel("SHAP Value")
    ax3.grid(True)
    ax3.text(
        0.5,
        -0.2,
        "Shows contribution of each feature\nto individual predictions",
        ha="center",
        va="top",
        transform=ax3.transAxes,
        fontsize=10,
    )

    # Decision Tree
    ax4 = plt.subplot(gs[1, 1])
    from sklearn.datasets import make_classification
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    X, y = make_classification(n_samples=100, n_features=2, n_classes=2)
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)
    plot_tree(clf, ax=ax4, filled=True)
    ax4.set_title("Decision Tree Visualization")
    ax4.text(
        0.5,
        -0.2,
        "Shows decision rules and splits\nin a tree-based model",
        ha="center",
        va="top",
        transform=ax4.transAxes,
        fontsize=10,
    )

    plt.tight_layout()
    save_figure("model_interpretation_guide.png")


if __name__ == "__main__":
    set_style()
    create_model_selection_flowchart()
    create_regularization_comparison()
    create_model_complexity_guide()
    create_logistic_regression_guide()
    create_polynomial_regression_guide()
    create_model_interpretation_guide()
