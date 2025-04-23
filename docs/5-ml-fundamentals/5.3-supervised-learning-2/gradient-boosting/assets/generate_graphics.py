import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use("default")
sns.set_theme(style="whitegrid")
sns.set_palette("husl")


def save_fig(name):
    """Save figure with consistent settings"""
    plt.savefig(f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


# 1. Introduction Graphics
def create_sequential_learning_diagram():
    """Create diagram showing sequential learning process"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create sequential learning diagram
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)

    # Plot initial prediction
    ax.plot(x, np.zeros_like(x), "r--", label="Initial Prediction")

    # Plot sequential improvements
    for i in range(1, 6):
        improvement = np.sin(x) * (i / 5)
        ax.plot(x, improvement, label=f"Model {i}")

    # Add final prediction
    ax.plot(x, np.sin(x), "k-", linewidth=2, label="Final Prediction")

    ax.set_title("Sequential Learning in Gradient Boosting")
    ax.set_xlabel("Input Features")
    ax.set_ylabel("Prediction")
    ax.legend()
    save_fig("sequential_learning")


def create_ensemble_diagram():
    """Create diagram showing ensemble of weak learners"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create ensemble diagram
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Plot individual weak learners
    for i in range(5):
        offset = np.random.normal(0, 0.2)
        ax.plot(x, y + offset, alpha=0.3, label=f"Weak Learner {i+1}")

    # Plot combined prediction
    ax.plot(x, y, "k-", linewidth=2, label="Combined Prediction")

    ax.set_title("Ensemble of Weak Learners")
    ax.set_xlabel("Input Features")
    ax.set_ylabel("Prediction")
    ax.legend()
    save_fig("ensemble_learners")


# 2. Math Foundation Graphics
def create_gradient_descent_diagram():
    """Create diagram showing gradient descent in function space"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create loss surface
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    # Plot contour
    contour = ax.contour(X, Y, Z, levels=20, cmap="viridis")

    # Plot gradient descent path
    path_x = np.linspace(-4, 0, 10)
    path_y = np.linspace(-4, 0, 10)
    ax.plot(path_x, path_y, "ro-", label="Gradient Descent Path")

    ax.set_title("Gradient Descent in Function Space")
    ax.set_xlabel("Parameter 1")
    ax.set_ylabel("Parameter 2")
    ax.legend()
    save_fig("gradient_descent")


def create_residual_learning_diagram():
    """Create diagram showing residual learning process"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create data
    x = np.linspace(0, 10, 100)
    y_true = np.sin(x)
    y_pred = np.zeros_like(x)

    # Plot true values and initial prediction
    ax.plot(x, y_true, "b-", label="True Values")
    ax.plot(x, y_pred, "r--", label="Initial Prediction")

    # Plot residuals
    residuals = y_true - y_pred
    ax.vlines(x, y_pred, y_true, colors="g", alpha=0.3, label="Residuals")

    ax.set_title("Residual Learning Process")
    ax.set_xlabel("Input Features")
    ax.set_ylabel("Value")
    ax.legend()
    save_fig("residual_learning")


# 3. Implementation Graphics
def create_feature_importance_diagram():
    """Create diagram showing feature importance"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create sample feature importance data
    features = ["Feature A", "Feature B", "Feature C", "Feature D", "Feature E"]
    importance = np.random.uniform(0, 1, len(features))
    importance = importance / importance.sum()

    # Plot feature importance
    sns.barplot(x=importance, y=features, ax=ax)

    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")
    save_fig("feature_importance")


def create_learning_curve():
    """Create diagram showing learning curve"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create sample learning curve data
    iterations = np.arange(1, 101)
    train_loss = np.exp(-iterations / 20) + np.random.normal(0, 0.01, 100)
    val_loss = np.exp(-iterations / 25) + np.random.normal(0, 0.01, 100)

    # Plot learning curves
    ax.plot(iterations, train_loss, label="Training Loss")
    ax.plot(iterations, val_loss, label="Validation Loss")

    ax.set_title("Learning Curve")
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("Loss")
    ax.legend()
    save_fig("learning_curve")


# 4. Advanced Topics Graphics
def create_shap_values_diagram():
    """Create diagram showing SHAP values"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create sample SHAP values
    features = ["Feature A", "Feature B", "Feature C", "Feature D", "Feature E"]
    shap_values = np.random.normal(0, 0.5, len(features))

    # Plot SHAP values
    colors = ["red" if x < 0 else "blue" for x in shap_values]
    ax.barh(features, shap_values, color=colors)

    ax.set_title("SHAP Values")
    ax.set_xlabel("Impact on Prediction")
    ax.set_ylabel("Features")
    save_fig("shap_values")


def create_partial_dependence_diagram():
    """Create diagram showing partial dependence"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create sample partial dependence data
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + 0.1 * x

    # Plot partial dependence
    ax.plot(x, y)
    ax.fill_between(x, y - 0.2, y + 0.2, alpha=0.2)

    ax.set_title("Partial Dependence Plot")
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Prediction")
    save_fig("partial_dependence")


# 5. Applications Graphics
def create_credit_risk_diagram():
    """Create diagram showing credit risk assessment"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create sample credit risk data
    scores = np.random.normal(600, 100, 1000)
    default = np.random.binomial(1, 1 / (1 + np.exp(-(scores - 600) / 100)))

    # Plot credit scores vs default probability
    sns.kdeplot(data=scores[default == 0], label="Non-Default", ax=ax)
    sns.kdeplot(data=scores[default == 1], label="Default", ax=ax)

    ax.set_title("Credit Score Distribution")
    ax.set_xlabel("Credit Score")
    ax.set_ylabel("Density")
    ax.legend()
    save_fig("credit_risk")


def create_churn_prediction_diagram():
    """Create diagram showing customer churn prediction"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create sample churn data
    tenure = np.random.exponential(20, 1000)
    churn = np.random.binomial(1, 1 / (1 + np.exp(-(tenure - 20) / 10)))

    # Plot tenure vs churn probability
    sns.kdeplot(data=tenure[churn == 0], label="Retained", ax=ax)
    sns.kdeplot(data=tenure[churn == 1], label="Churned", ax=ax)

    ax.set_title("Customer Tenure Distribution")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Density")
    ax.legend()
    save_fig("churn_prediction")


# Generate all graphics
if __name__ == "__main__":
    # Introduction graphics
    create_sequential_learning_diagram()
    create_ensemble_diagram()

    # Math foundation graphics
    create_gradient_descent_diagram()
    create_residual_learning_diagram()

    # Implementation graphics
    create_feature_importance_diagram()
    create_learning_curve()

    # Advanced topics graphics
    create_shap_values_diagram()
    create_partial_dependence_diagram()

    # Applications graphics
    create_credit_risk_diagram()
    create_churn_prediction_diagram()
