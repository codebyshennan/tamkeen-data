import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier

# Create assets directory if it doesn't exist
os.makedirs("assets", exist_ok=True)


def save_plot(filename):
    """Helper function to save plots"""
    plt.savefig(f"assets/{filename}", dpi=300, bbox_inches="tight")
    plt.close()


# 1. Introduction Visualization: KNN Decision Boundary
def create_knn_decision_boundary():
    # Generate synthetic data
    X, y = make_blobs(n_samples=100, centers=2, random_state=42)

    # Create mesh grid
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)

    # Predict for mesh points
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
    plt.title("KNN Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    save_plot("knn_decision_boundary.png")


# 2. Distance Metrics Visualization
def create_distance_metrics_plot():
    # Create sample points
    points = np.array([[0, 0], [3, 4]])

    plt.figure(figsize=(12, 4))

    # Euclidean Distance
    plt.subplot(131)
    plt.plot([points[0, 0], points[1, 0]], [points[0, 1], points[1, 1]], "r-")
    plt.scatter(points[:, 0], points[:, 1], c="b", s=50)
    plt.title("Euclidean Distance")
    plt.grid(True)

    # Manhattan Distance
    plt.subplot(132)
    plt.plot(
        [points[0, 0], points[1, 0], points[1, 0]],
        [points[0, 1], points[0, 1], points[1, 1]],
        "g-",
    )
    plt.scatter(points[:, 0], points[:, 1], c="b", s=50)
    plt.title("Manhattan Distance")
    plt.grid(True)

    # Cosine Similarity
    plt.subplot(133)
    plt.arrow(
        0,
        0,
        points[1, 0],
        points[1, 1],
        head_width=0.5,
        head_length=0.5,
        fc="b",
        ec="b",
    )
    plt.xlim(-1, 4)
    plt.ylim(-1, 5)
    plt.title("Cosine Similarity")
    plt.grid(True)

    plt.tight_layout()
    save_plot("distance_metrics.png")


# 3. Implementation Visualization: KNN with Different k Values
def create_k_values_plot():
    # Generate synthetic data
    X, y = make_blobs(n_samples=100, centers=3, random_state=42)

    plt.figure(figsize=(15, 5))

    # Plot for different k values
    for i, k in enumerate([1, 5, 15]):
        plt.subplot(1, 3, i + 1)

        # Train KNN
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, y)

        # Create mesh grid
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Predict
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
        plt.title(f"KNN with k={k}")

    plt.tight_layout()
    save_plot("knn_different_k.png")


# 4. Advanced Techniques: Weighted KNN
def create_weighted_knn_plot():
    # Generate synthetic data
    X, y = make_blobs(n_samples=100, centers=2, random_state=42)

    plt.figure(figsize=(12, 5))

    # Plot uniform weights
    plt.subplot(121)
    knn = KNeighborsClassifier(n_neighbors=5, weights="uniform")
    knn.fit(X, y)

    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
    plt.title("Uniform Weights")

    # Plot distance weights
    plt.subplot(122)
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    knn.fit(X, y)

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
    plt.title("Distance Weights")

    plt.tight_layout()
    save_plot("weighted_knn.png")


# 5. Applications: Movie Recommendation Visualization
def create_recommendation_plot():
    # Create synthetic movie ratings
    np.random.seed(42)
    ratings = np.random.randint(1, 6, size=(10, 5))

    plt.figure(figsize=(10, 6))
    plt.imshow(ratings, cmap="viridis")
    plt.colorbar(label="Rating")
    plt.title("Movie Ratings Matrix")
    plt.xlabel("Users")
    plt.ylabel("Movies")
    plt.xticks(range(5), [f"User {i+1}" for i in range(5)])
    plt.yticks(range(10), [f"Movie {i+1}" for i in range(10)])
    save_plot("movie_ratings.png")


# Generate all visualizations
create_knn_decision_boundary()
create_distance_metrics_plot()
create_k_values_plot()
create_weighted_knn_plot()
create_recommendation_plot()
