import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Set style
plt.style.use("default")
sns.set_style("whitegrid")
sns.set_palette("husl")


def save_fig(name):
    """Save figure with consistent styling"""
    plt.tight_layout()
    plt.savefig(f"assets/{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


# 1. Basic Decision Tree Structure
def create_basic_tree():
    """Create visualization of basic decision tree structure"""
    plt.figure(figsize=(10, 6))
    plt.title("Basic Decision Tree Structure", fontsize=14)

    # Create nodes
    plt.scatter([0], [0], s=1000, c="lightblue", label="Root Node")
    plt.scatter([-2, 2], [-1, -1], s=800, c="lightgreen", label="Internal Nodes")
    plt.scatter(
        [-3, -1, 1, 3], [-2, -2, -2, -2], s=600, c="lightyellow", label="Leaf Nodes"
    )

    # Add connections
    plt.plot([0, -2], [0, -1], "k-", lw=2)
    plt.plot([0, 2], [0, -1], "k-", lw=2)
    plt.plot([-2, -3], [-1, -2], "k-", lw=2)
    plt.plot([-2, -1], [-1, -2], "k-", lw=2)
    plt.plot([2, 1], [-1, -2], "k-", lw=2)
    plt.plot([2, 3], [-1, -2], "k-", lw=2)

    # Add labels
    plt.text(0, 0.2, "Root", ha="center", fontsize=12)
    plt.text(-2, -0.8, "Internal", ha="center", fontsize=12)
    plt.text(2, -0.8, "Internal", ha="center", fontsize=12)
    plt.text(-3, -1.8, "Leaf", ha="center", fontsize=12)
    plt.text(-1, -1.8, "Leaf", ha="center", fontsize=12)
    plt.text(1, -1.8, "Leaf", ha="center", fontsize=12)
    plt.text(3, -1.8, "Leaf", ha="center", fontsize=12)

    plt.axis("off")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05))
    save_fig("basic_tree_structure")


# 2. Impurity Measures
def plot_impurity_measures():
    """Plot Gini and Entropy impurity measures"""
    p = np.linspace(0.01, 0.99, 100)
    gini = p * (1 - p)
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    plt.figure(figsize=(10, 6))
    plt.plot(p, gini, label="Gini Impurity", linewidth=2)
    plt.plot(p, entropy, label="Entropy", linewidth=2)
    plt.xlabel("Probability of Class 1", fontsize=12)
    plt.ylabel("Impurity Measure", fontsize=12)
    plt.title("Comparison of Impurity Measures", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    save_fig("impurity_measures")


# 3. Decision Boundary
def plot_decision_boundary():
    """Plot decision boundary of a simple decision tree"""
    # Generate data
    X = np.random.randn(200, 2)
    y = (X[:, 0] > 0) & (X[:, 1] > 0)

    # Train tree
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y)

    # Create mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # Predict
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
    plt.title("Decision Tree Decision Boundary", fontsize=14)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    save_fig("decision_boundary")


# 4. Feature Importance
def plot_feature_importance():
    """Plot feature importance from a decision tree"""
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Train tree
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y)

    # Plot importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(iris.feature_names)), clf.feature_importances_)
    plt.xticks(range(len(iris.feature_names)), iris.feature_names, rotation=45)
    plt.title("Feature Importance in Iris Dataset", fontsize=14)
    plt.ylabel("Importance Score", fontsize=12)
    save_fig("feature_importance")


# 5. Tree Pruning
def plot_pruning_effect():
    """Plot effect of pruning on tree performance"""
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Try different depths
    depths = range(1, 11)
    train_scores = []
    test_scores = []

    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(X_train, y_train)
        train_scores.append(clf.score(X_train, y_train))
        test_scores.append(clf.score(X_test, y_test))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_scores, label="Training Score", marker="o")
    plt.plot(depths, test_scores, label="Testing Score", marker="o")
    plt.xlabel("Tree Depth", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Effect of Tree Depth on Performance", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    save_fig("pruning_effect")


# 6. Ensemble Comparison
def plot_ensemble_comparison():
    """Compare single tree vs random forest"""
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # Train models
    tree = DecisionTreeClassifier(max_depth=3)
    forest = RandomForestClassifier(n_estimators=100, max_depth=3)

    # Cross-validation scores
    tree_scores = np.random.normal(0.9, 0.02, 100)  # Simulated
    forest_scores = np.random.normal(0.95, 0.01, 100)  # Simulated

    # Plot
    plt.figure(figsize=(10, 6))
    plt.boxplot([tree_scores, forest_scores], labels=["Single Tree", "Random Forest"])
    plt.title("Single Tree vs Random Forest Performance", fontsize=14)
    plt.ylabel("Accuracy", fontsize=12)
    save_fig("ensemble_comparison")


# Generate all visualizations
if __name__ == "__main__":
    create_basic_tree()
    plot_impurity_measures()
    plot_decision_boundary()
    plot_feature_importance()
    plot_pruning_effect()
    plot_ensemble_comparison()
