import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.preprocessing import StandardScaler

# Create assets directory if it doesn't exist
os.makedirs("assets", exist_ok=True)

# Set style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


# 1. Regularization Path Diagram
def plot_regularization_path():
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.random.randn(100)

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create regularization path
    alphas = np.logspace(-3, 3, 100)
    coefs_ridge = []
    coefs_lasso = []

    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        lasso = Lasso(alpha=alpha)
        ridge.fit(X_scaled, y)
        lasso.fit(X_scaled, y)
        coefs_ridge.append(ridge.coef_)
        coefs_lasso.append(lasso.coef_)

    coefs_ridge = np.array(coefs_ridge)
    coefs_lasso = np.array(coefs_lasso)

    # Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for i in range(3):
        plt.plot(np.log10(alphas), coefs_ridge[:, i], label=f"Feature {i+1}")
    plt.xlabel("log(alpha)")
    plt.ylabel("Coefficient Value")
    plt.title("Ridge Regularization Path")
    plt.legend()

    plt.subplot(1, 2, 2)
    for i in range(3):
        plt.plot(np.log10(alphas), coefs_lasso[:, i], label=f"Feature {i+1}")
    plt.xlabel("log(alpha)")
    plt.ylabel("Coefficient Value")
    plt.title("Lasso Regularization Path")
    plt.legend()

    plt.tight_layout()
    plt.savefig("assets/regularization_path.png", dpi=300, bbox_inches="tight")
    plt.close()


# 2. Bias-Variance Tradeoff
def plot_bias_variance_tradeoff():
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 20)
    y = np.random.randn(100)

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Calculate bias and variance for different alpha values
    alphas = np.logspace(-3, 3, 50)
    biases = []
    variances = []

    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_scaled, y)
        y_pred = ridge.predict(X_scaled)

        bias = np.mean((y - y_pred) ** 2)
        variance = np.var(y_pred)

        biases.append(bias)
        variances.append(variance)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(np.log10(alphas), biases, label="Bias²", linewidth=2)
    plt.plot(np.log10(alphas), variances, label="Variance", linewidth=2)
    plt.plot(
        np.log10(alphas),
        np.array(biases) + np.array(variances),
        label="Total Error",
        linewidth=2,
        linestyle="--",
    )

    plt.xlabel("log(alpha)")
    plt.ylabel("Error")
    plt.title("Bias-Variance Tradeoff")
    plt.legend()
    plt.grid(True)

    plt.savefig("assets/bias_variance_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close()


# 3. Feature Selection Comparison
def plot_feature_selection():
    # Generate sample data with some irrelevant features
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit different models
    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=0.1)
    elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)

    ridge.fit(X_scaled, y)
    lasso.fit(X_scaled, y)
    elastic.fit(X_scaled, y)

    # Plot coefficients
    plt.figure(figsize=(12, 6))
    x = np.arange(10)
    width = 0.25

    plt.bar(x - width, ridge.coef_, width, label="Ridge")
    plt.bar(x, lasso.coef_, width, label="Lasso")
    plt.bar(x + width, elastic.coef_, width, label="Elastic Net")

    plt.xlabel("Features")
    plt.ylabel("Coefficient Value")
    plt.title("Feature Selection Comparison")
    plt.legend()
    plt.xticks(x, [f"Feature {i+1}" for i in range(10)])

    plt.savefig("assets/feature_selection.png", dpi=300, bbox_inches="tight")
    plt.close()


# 4. Overfitting Prevention
def plot_overfitting_prevention():
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 20)
    y = X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train = X_scaled[:80]
    y_train = y[:80]
    X_test = X_scaled[80:]
    y_test = y[80:]

    # Calculate training and testing errors for different alpha values
    alphas = np.logspace(-3, 3, 50)
    train_errors = []
    test_errors = []

    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)

        train_pred = ridge.predict(X_train)
        test_pred = ridge.predict(X_test)

        train_errors.append(np.mean((y_train - train_pred) ** 2))
        test_errors.append(np.mean((y_test - test_pred) ** 2))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(np.log10(alphas), train_errors, label="Training Error", linewidth=2)
    plt.plot(np.log10(alphas), test_errors, label="Testing Error", linewidth=2)

    plt.xlabel("log(alpha)")
    plt.ylabel("Mean Squared Error")
    plt.title("Overfitting Prevention with Regularization")
    plt.legend()
    plt.grid(True)

    plt.savefig("assets/overfitting_prevention.png", dpi=300, bbox_inches="tight")
    plt.close()


# Generate all diagrams
if __name__ == "__main__":
    plot_regularization_path()
    plot_bias_variance_tradeoff()
    plot_feature_selection()
    plot_overfitting_prevention()
    print("Diagrams generated successfully in the assets folder!")
