import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import learning_curve


def set_style():
    """Set consistent style for all visualizations"""
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    plt.rcParams["figure.figsize"] = [10, 6]
    plt.rcParams["font.size"] = 12


def plot_ml_paradigm():
    """Visualize the difference between traditional programming and ML"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Traditional Programming
    ax1.set_title("Traditional Programming")
    rect1 = plt.Rectangle((0.1, 0.3), 0.2, 0.4, fc="lightblue", label="Input")
    rect2 = plt.Rectangle((0.4, 0.3), 0.2, 0.4, fc="lightgreen", label="Rules")
    rect3 = plt.Rectangle((0.7, 0.3), 0.2, 0.4, fc="lightcoral", label="Output")

    for rect in [rect1, rect2, rect3]:
        ax1.add_patch(rect)

    ax1.annotate("", xy=(0.3, 0.5), xytext=(0.4, 0.5), arrowprops=dict(arrowstyle="->"))
    ax1.annotate("", xy=(0.6, 0.5), xytext=(0.7, 0.5), arrowprops=dict(arrowstyle="->"))

    # Machine Learning
    ax2.set_title("Machine Learning")
    rect4 = plt.Rectangle((0.1, 0.3), 0.2, 0.4, fc="lightblue", label="Input + Output")
    rect5 = plt.Rectangle((0.4, 0.3), 0.2, 0.4, fc="lightgreen", label="ML Algorithm")
    rect6 = plt.Rectangle((0.7, 0.3), 0.2, 0.4, fc="lightcoral", label="Rules (Model)")

    for rect in [rect4, rect5, rect6]:
        ax2.add_patch(rect)

    ax2.annotate("", xy=(0.3, 0.5), xytext=(0.4, 0.5), arrowprops=dict(arrowstyle="->"))
    ax2.annotate("", xy=(0.6, 0.5), xytext=(0.7, 0.5), arrowprops=dict(arrowstyle="->"))

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    plt.tight_layout()
    return fig


def plot_supervised_learning():
    """Visualize supervised learning with regression and classification examples"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Regression Example
    X_reg, y_reg = make_regression(
        n_samples=100, n_features=1, noise=10, random_state=42
    )
    model_reg = LinearRegression()
    model_reg.fit(X_reg, y_reg)

    ax1.scatter(X_reg, y_reg, alpha=0.5, label="Data Points")
    ax1.plot(X_reg, model_reg.predict(X_reg), color="red", label="Regression Line")
    ax1.set_title("Regression Example")
    ax1.set_xlabel("Feature")
    ax1.set_ylabel("Target")
    ax1.legend()

    # Classification Example
    X_clf, y_clf = make_classification(
        n_samples=100,
        n_features=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )
    model_clf = LogisticRegression()
    model_clf.fit(X_clf, y_clf)

    ax2.scatter(X_clf[:, 0], X_clf[:, 1], c=y_clf, cmap="coolwarm", alpha=0.5)
    ax2.set_title("Classification Example")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")

    plt.tight_layout()
    return fig


def plot_unsupervised_learning():
    """Visualize unsupervised learning with clustering example"""
    # Generate sample data
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

    # Fit KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    y_pred = kmeans.fit_predict(X)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="viridis", alpha=0.6)
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        color="red",
        marker="x",
        s=200,
        linewidth=3,
        label="Centroids",
    )
    plt.title("Unsupervised Learning: Clustering Example")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    return plt.gcf()


def plot_ml_workflow():
    """Create a visual representation of the ML workflow"""
    G = nx.DiGraph()

    # Add nodes with positions
    nodes = [
        "Problem\nDefinition",
        "Data\nCollection",
        "Data\nPreparation",
        "Model\nSelection",
        "Model\nTraining",
        "Model\nEvaluation",
        "Model\nDeployment",
    ]

    # Create circular layout
    pos = nx.circular_layout(nodes)

    # Add nodes and edges
    for i, node in enumerate(nodes):
        G.add_node(node)
        if i > 0:
            G.add_edge(nodes[i - 1], node)

    # Add final edge to close the cycle
    G.add_edge(nodes[-1], nodes[0])

    # Draw
    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos=pos,
        with_labels=True,
        node_color="lightblue",
        node_size=3000,
        font_size=10,
        font_weight="bold",
        arrows=True,
        edge_color="gray",
        arrowsize=20,
    )

    plt.title("Machine Learning Workflow", pad=20)
    return plt.gcf()


def plot_bias_variance():
    """Visualize bias-variance tradeoff"""
    np.random.seed(42)

    # Generate true function
    X = np.linspace(0, 10, 100)
    y_true = np.sin(X) + 1

    # Generate noisy data
    X_data = np.random.choice(X, 20)
    y_data = np.sin(X_data) + 1 + np.random.normal(0, 0.2, 20)

    # Different complexity models
    plt.figure(figsize=(15, 5))

    # Underfitting (High Bias)
    plt.subplot(131)
    plt.scatter(X_data, y_data, color="blue", alpha=0.5, label="Data")
    plt.plot(X, y_true, "g--", label="True Function")
    plt.plot(X, np.poly1d(np.polyfit(X_data, y_data, 1))(X), "r-", label="Model")
    plt.title("Underfitting (High Bias)")
    plt.legend()

    # Good Fit
    plt.subplot(132)
    plt.scatter(X_data, y_data, color="blue", alpha=0.5, label="Data")
    plt.plot(X, y_true, "g--", label="True Function")
    plt.plot(X, np.poly1d(np.polyfit(X_data, y_data, 3))(X), "r-", label="Model")
    plt.title("Good Fit")
    plt.legend()

    # Overfitting (High Variance)
    plt.subplot(133)
    plt.scatter(X_data, y_data, color="blue", alpha=0.5, label="Data")
    plt.plot(X, y_true, "g--", label="True Function")
    plt.plot(X, np.poly1d(np.polyfit(X_data, y_data, 15))(X), "r-", label="Model")
    plt.title("Overfitting (High Variance)")
    plt.legend()

    plt.tight_layout()
    return plt.gcf()


def plot_learning_curves_example():
    """Visualize learning curves for understanding model performance"""
    # Generate sample data
    X, y = make_regression(n_samples=300, n_features=1, noise=10, random_state=42)

    # Calculate learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        LinearRegression(), X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), "o-", label="Training Score")
    plt.plot(
        train_sizes, np.mean(val_scores, axis=1), "o-", label="Cross-validation Score"
    )
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.title("Learning Curves")
    plt.legend(loc="best")
    plt.grid(True)

    return plt.gcf()


if __name__ == "__main__":
    # Set style for all plots
    set_style()

    # Generate and save all visualizations
    visualizations = {
        "ml_paradigm": plot_ml_paradigm(),
        "supervised_learning": plot_supervised_learning(),
        "unsupervised_learning": plot_unsupervised_learning(),
        "ml_workflow": plot_ml_workflow(),
        "bias_variance": plot_bias_variance(),
        "learning_curves": plot_learning_curves_example(),
    }

    # Save all figures
    for name, fig in visualizations.items():
        fig.savefig(
            f"docs/5-ml-fundamentals/5.1-intro-to-ml/images/{name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
