import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.svm import SVC


def save_fig(name):
    """Save figure to assets folder"""
    plt.savefig(f"assets/{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


# 1. Introduction Visualizations
def create_intro_visualizations():
    # Create linearly separable data
    X, y = make_blobs(n_samples=100, centers=2, random_state=42)

    # Train SVM
    clf = SVC(kernel="linear")
    clf.fit(X, y)

    # Plot decision boundary
    plt.figure(figsize=(10, 6))

    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(["#FF9999", "#9999FF"]))

    # Plot support vectors
    plt.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )

    # Plot decision boundary
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # Plot decision boundary and margins
    ax.contour(
        XX,
        YY,
        Z,
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
    )

    plt.title("SVM Decision Boundary and Support Vectors")
    save_fig("svm_decision_boundary")


# 2. Math and Kernels Visualizations
def create_kernel_visualizations():
    # Create non-linear data
    X, y = make_circles(n_samples=100, factor=0.3, noise=0.1, random_state=42)

    # Create subplot for different kernels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    kernels = ["linear", "rbf", "poly"]
    titles = ["Linear Kernel", "RBF Kernel", "Polynomial Kernel"]

    for ax, kernel, title in zip(axes, kernels, titles):
        # Train SVM
        clf = SVC(kernel=kernel)
        clf.fit(X, y)

        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02)
        )

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.4)
        ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        ax.set_title(title)

    plt.tight_layout()
    save_fig("kernel_comparison")


# 3. Implementation Visualizations
def create_implementation_visualizations():
    # Create imbalanced data
    X1, y1 = make_blobs(n_samples=20, centers=1, random_state=42)
    X2, y2 = make_blobs(
        n_samples=100, centers=1, cluster_std=1.5, center_box=(3, 3), random_state=42
    )
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros(20), np.ones(100)))

    # Create subplot for different class weights
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    weights = [None, "balanced"]
    titles = ["Without Class Weights", "With Class Weights"]

    for ax, weight, title in zip(axes, weights, titles):
        # Train SVM
        clf = SVC(kernel="linear", class_weight=weight)
        clf.fit(X, y)

        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02)
        )

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.4)
        ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        ax.set_title(title)

    plt.tight_layout()
    save_fig("class_weights_comparison")


# 4. Advanced Techniques Visualizations
def create_advanced_visualizations():
    # Create complex data
    X, y = make_moons(n_samples=100, noise=0.2, random_state=42)

    # Create subplot for different C values
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    C_values = [0.1, 1, 10]
    titles = [f"C = {C}" for C in C_values]

    for ax, C, title in zip(axes, C_values, titles):
        # Train SVM
        clf = SVC(kernel="rbf", C=C)
        clf.fit(X, y)

        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02)
        )

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.4)
        ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        ax.set_title(title)

    plt.tight_layout()
    save_fig("C_parameter_comparison")


# 5. Applications Visualizations
def create_applications_visualizations():
    # Create synthetic data for different applications
    # Text classification
    X_text = np.random.rand(100, 2)
    y_text = (X_text[:, 0] + X_text[:, 1] > 1).astype(int)

    # Image classification
    X_img = np.random.rand(100, 2)
    y_img = ((X_img[:, 0] - 0.5) ** 2 + (X_img[:, 1] - 0.5) ** 2 < 0.2).astype(int)

    # Medical diagnosis
    X_med = np.random.rand(100, 2)
    y_med = (X_med[:, 0] > 0.5).astype(int)

    # Create subplot for different applications
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    datasets = [(X_text, y_text), (X_img, y_img), (X_med, y_med)]
    titles = ["Text Classification", "Image Recognition", "Medical Diagnosis"]

    for ax, (X, y), title in zip(axes, datasets, titles):
        # Train SVM
        clf = SVC(kernel="rbf")
        clf.fit(X, y)

        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02)
        )

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.4)
        ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        ax.set_title(title)

    plt.tight_layout()
    save_fig("applications_comparison")


if __name__ == "__main__":
    # Generate all visualizations
    create_intro_visualizations()
    create_kernel_visualizations()
    create_implementation_visualizations()
    create_advanced_visualizations()
    create_applications_visualizations()
    create_applications_visualizations()
