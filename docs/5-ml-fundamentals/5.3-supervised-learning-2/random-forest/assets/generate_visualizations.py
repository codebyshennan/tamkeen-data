import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Set style
plt.style.use("seaborn")
sns.set_palette("husl")


def create_decision_tree_visualization():
    """Create a visualization of how a decision tree makes decisions"""
    # Create sample data
    X, y = make_classification(
        n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42
    )

    # Create and train a simple decision tree
    rf = RandomForestClassifier(n_estimators=1, max_depth=3, random_state=42)
    rf.fit(X, y)

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # Plot decision boundary
    plt.figure(figsize=(10, 8))
    Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title("Decision Tree Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig("decision_tree_boundary.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_random_forest_visualization():
    """Create a visualization of how multiple trees work together"""
    # Create sample data
    X, y = make_classification(
        n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42
    )

    # Create and train a random forest
    rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    rf.fit(X, y)

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # Plot decision boundary
    plt.figure(figsize=(10, 8))
    Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title("Random Forest Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig("random_forest_boundary.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_feature_importance_visualization():
    """Create a visualization of feature importance"""
    # Create sample data
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=42
    )

    # Create and train a random forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), [f"Feature {i+1}" for i in indices], rotation=45)
    plt.title("Feature Importance in Random Forest")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_bias_variance_visualization():
    """Create a visualization of bias-variance tradeoff"""
    # Create sample data
    X = np.linspace(0, 10, 100)
    y = np.sin(X) + np.random.normal(0, 0.1, 100)

    # Create and train models with different complexities
    models = []
    for depth in [1, 3, 5, 10]:
        rf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
        rf.fit(X.reshape(-1, 1), (y > 0).astype(int))
        models.append(rf)

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.scatter(X, y, alpha=0.5, label="Data")

    for i, model in enumerate(models):
        y_pred = model.predict_proba(X.reshape(-1, 1))[:, 1]
        plt.plot(X, y_pred, label=f"Depth {[1, 3, 5, 10][i]}")

    plt.title("Bias-Variance Tradeoff in Random Forest")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.savefig("bias_variance.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_ensemble_visualization():
    """Create a visualization of how ensemble predictions work"""
    # Create sample data
    X = np.linspace(0, 10, 100)
    y = np.sin(X) + np.random.normal(0, 0.2, 100)

    # Create and train multiple trees
    trees = []
    for i in range(5):
        rf = RandomForestClassifier(n_estimators=1, max_depth=3, random_state=i)
        rf.fit(X.reshape(-1, 1), (y > 0).astype(int))
        trees.append(rf)

    # Plot individual tree predictions and ensemble
    plt.figure(figsize=(12, 8))
    plt.scatter(X, y, alpha=0.5, label="Data")

    # Plot individual tree predictions
    for i, tree in enumerate(trees):
        y_pred = tree.predict_proba(X.reshape(-1, 1))[:, 1]
        plt.plot(X, y_pred, alpha=0.3, label=f"Tree {i+1}")

    # Plot ensemble prediction
    y_ensemble = np.mean(
        [tree.predict_proba(X.reshape(-1, 1))[:, 1] for tree in trees], axis=0
    )
    plt.plot(X, y_ensemble, "k--", linewidth=2, label="Ensemble")

    plt.title("Individual Trees vs Ensemble Prediction")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.savefig("ensemble_prediction.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Generate all visualizations"""
    create_decision_tree_visualization()
    create_random_forest_visualization()
    create_feature_importance_visualization()
    create_bias_variance_visualization()
    create_ensemble_visualization()


if __name__ == "__main__":
    main()
