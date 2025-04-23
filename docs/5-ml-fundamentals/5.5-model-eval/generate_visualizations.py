import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Set style
plt.style.use("seaborn-v0_8")
sns.set_theme()

# Create assets directory if it doesn't exist
import os

if not os.path.exists("assets"):
    os.makedirs("assets")

# Set the random seed for reproducibility
np.random.seed(42)


# 1. Cross Validation Visualizations
def generate_cv_visualizations():
    """Generate visualizations for cross-validation techniques."""

    # Generate sample data
    X = np.arange(100).reshape(100, 1)
    y = np.random.randint(0, 2, 100)

    # K-Fold visualization
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    plt.figure(figsize=(12, 4))
    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        plt.scatter(
            X[train_idx],
            [i] * len(train_idx),
            c="blue",
            alpha=0.6,
            label="Training" if i == 0 else "",
        )
        plt.scatter(
            X[val_idx],
            [i] * len(val_idx),
            c="red",
            alpha=0.6,
            label="Validation" if i == 0 else "",
        )
    plt.yticks(range(5), [f"Fold {i+1}" for i in range(5)])
    plt.xlabel("Sample Index")
    plt.title("K-Fold Cross Validation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/kfold_visualization.png")
    plt.close()

    # Stratified K-Fold vs Regular K-Fold
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Regular K-Fold
    for i, (_, val_idx) in enumerate(kf.split(X)):
        val_dist = np.bincount(y[val_idx]) / len(val_idx)
        ax1.bar(np.arange(2) + i * 3, val_dist, alpha=0.6)
    ax1.set_title("Class Distribution in Regular K-Fold")
    ax1.set_xticks(np.arange(0, 15, 3))
    ax1.set_xticklabels([f"Fold {i+1}" for i in range(5)])

    # Stratified K-Fold
    for i, (_, val_idx) in enumerate(skf.split(X, y)):
        val_dist = np.bincount(y[val_idx]) / len(val_idx)
        ax2.bar(np.arange(2) + i * 3, val_dist, alpha=0.6)
    ax2.set_title("Class Distribution in Stratified K-Fold")
    ax2.set_xticks(np.arange(0, 15, 3))
    ax2.set_xticklabels([f"Fold {i+1}" for i in range(5)])

    plt.tight_layout()
    plt.savefig("assets/stratified_vs_regular_kfold.png")
    plt.close()

    # Time Series Cross Validation
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=5)

    plt.figure(figsize=(12, 4))
    for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
        plt.scatter(
            X[train_idx],
            [i] * len(train_idx),
            c="blue",
            alpha=0.6,
            label="Training" if i == 0 else "",
        )
        plt.scatter(
            X[val_idx],
            [i] * len(val_idx),
            c="red",
            alpha=0.6,
            label="Validation" if i == 0 else "",
        )
    plt.yticks(range(5), [f"Split {i+1}" for i in range(5)])
    plt.xlabel("Time")
    plt.title("Time Series Cross Validation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/timeseries_cv.png")
    plt.close()


# 2. Hyperparameter Tuning Visualizations
def generate_hp_tuning_visualizations():
    """Generate visualizations for hyperparameter tuning."""

    # Generate sample hyperparameter tuning results
    n_samples = 100
    param_range = np.linspace(0.1, 10, n_samples)

    # Grid Search
    grid_scores = -0.5 * (param_range - 5) ** 2 + np.random.normal(0, 2, n_samples)
    grid_scores = (grid_scores - grid_scores.min()) / (
        grid_scores.max() - grid_scores.min()
    )

    plt.figure(figsize=(10, 6))
    plt.plot(param_range, grid_scores, "bo-", alpha=0.6)
    plt.xlabel("Hyperparameter Value")
    plt.ylabel("Validation Score")
    plt.title("Grid Search Results")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("assets/grid_search_results.png")
    plt.close()

    # Random Search
    random_params = np.random.uniform(0.1, 10, 30)
    random_scores = -0.5 * (random_params - 5) ** 2 + np.random.normal(0, 2, 30)
    random_scores = (random_scores - random_scores.min()) / (
        random_scores.max() - random_scores.min()
    )

    plt.figure(figsize=(10, 6))
    plt.scatter(random_params, random_scores, c="red", alpha=0.6)
    plt.xlabel("Hyperparameter Value")
    plt.ylabel("Validation Score")
    plt.title("Random Search Results")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("assets/random_search_results.png")
    plt.close()

    # Bayesian Optimization
    x = np.linspace(0, 10, 100)
    mean = -0.5 * (x - 5) ** 2
    mean = (mean - mean.min()) / (mean.max() - mean.min())
    std = 0.2 * np.exp(-0.5 * (x - 5) ** 2)

    plt.figure(figsize=(10, 6))
    plt.plot(x, mean, "b-", label="Mean")
    plt.fill_between(
        x, mean - 2 * std, mean + 2 * std, color="b", alpha=0.2, label="Uncertainty"
    )
    plt.scatter(x[::10], mean[::10], c="red", alpha=0.6, label="Observations")
    plt.xlabel("Hyperparameter Value")
    plt.ylabel("Objective")
    plt.title("Bayesian Optimization")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("assets/bayesian_optimization.png")
    plt.close()


# 3. Model Metrics Visualizations
def generate_metrics_visualizations():
    """Generate visualizations for model evaluation metrics."""

    # Generate classification dataset
    X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Confusion Matrix
    y_pred = (X[:, 0] > 0).astype(int)  # Simple threshold-based prediction
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("assets/confusion_matrix.png")
    plt.close()

    # ROC Curve
    y_score = X[:, 0]  # Using first feature as score for illustration
    fpr, tpr, _ = roc_curve(y, y_score)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, "b-", label="ROC curve")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("assets/roc_curve.png")
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y, y_score)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, "g-", label="PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("assets/precision_recall_curve.png")
    plt.close()

    # Regression Metrics
    X_reg, y_reg = make_regression(
        n_samples=100, n_features=1, noise=20, random_state=42
    )
    y_pred_reg = 0.9 * X_reg.ravel() + np.random.normal(0, 0.1, 100)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_reg, y_reg, c="blue", alpha=0.6, label="Actual")
    plt.scatter(X_reg, y_pred_reg, c="red", alpha=0.6, label="Predicted")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("Regression Predictions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("assets/regression_predictions.png")
    plt.close()

    # Residual Plot
    residuals = y_reg - y_pred_reg
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred_reg, residuals, c="blue", alpha=0.6)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("assets/residual_plot.png")
    plt.close()


# 4. Model Selection Visualizations
def generate_model_selection_visualizations():
    # Create sample dataset
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train models
    models = {
        "Linear": LogisticRegression(),
        "Tree": RandomForestClassifier(),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50)),
    }

    # Model Comparison
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = np.mean(cross_val_score(model, X, y, cv=5))

    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.title("Model Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("assets/model_comparison.png")
    plt.close()

    # Learning Curves
    for name, model in models.items():
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label="Training score")
        plt.plot(train_sizes, val_mean, label="Cross-validation score")
        plt.fill_between(
            train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1
        )
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
        plt.xlabel("Training Examples")
        plt.ylabel("Score")
        plt.title(f"Learning Curve - {name}")
        plt.legend(loc="best")
        plt.grid(True)
        plt.savefig(f'assets/learning_curve_{name.lower().replace(" ", "_")}.png')
        plt.close()


# Generate all visualizations
if __name__ == "__main__":
    print("Generating Cross Validation visualizations...")
    generate_cv_visualizations()

    print("Generating Hyperparameter Tuning visualizations...")
    generate_hp_tuning_visualizations()

    print("Generating Model Metrics visualizations...")
    generate_metrics_visualizations()

    print("Generating Model Selection visualizations...")
    generate_model_selection_visualizations()

    print("All visualizations have been generated and saved in the assets directory!")
