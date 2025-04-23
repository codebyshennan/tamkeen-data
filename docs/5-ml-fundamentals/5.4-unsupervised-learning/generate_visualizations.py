import os

import matplotlib.pyplot as plt
import numpy as np
import umap
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import load_digits, make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Create assets directory if it doesn't exist
os.makedirs("assets", exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)


# 1. PCA Visualizations
def generate_pca_visualizations():
    # Create sample data
    n_samples = 300
    t = np.random.uniform(0, 2 * np.pi, n_samples)
    x = np.cos(t) + np.random.normal(0, 0.1, n_samples)
    y = np.sin(t) + np.random.normal(0, 0.1, n_samples)
    data = np.column_stack((x, y))

    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Apply PCA
    pca = PCA()
    data_pca = pca.fit_transform(data_scaled)

    # Create visualization
    plt.figure(figsize=(15, 5))

    # Original data
    plt.subplot(131)
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], alpha=0.5)
    plt.title("Original Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # Data with principal components
    plt.subplot(132)
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], alpha=0.5)
    for i, (comp1, comp2) in enumerate(zip(pca.components_[0], pca.components_[1])):
        plt.arrow(
            0, 0, comp1, comp2, color="r", alpha=0.8, head_width=0.05, head_length=0.1
        )
    plt.title("Data with Principal Components")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # Transformed data
    plt.subplot(133)
    plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5)
    plt.title("Data in Principal Component Space")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")

    plt.tight_layout()
    plt.savefig("assets/pca_basic_example.png")
    plt.close()

    # Image compression example
    digits = load_digits()
    X = digits.data

    # Apply PCA with different numbers of components
    n_components_list = [10, 20, 50, 64]
    fig, axes = plt.subplots(2, len(n_components_list), figsize=(15, 6))

    # Original image
    sample_digit = X[0].reshape(8, 8)
    for ax in axes[0]:
        ax.imshow(sample_digit, cmap="gray")
        ax.axis("off")
        ax.set_title("Original")

    # Reconstructed images
    for i, n_comp in enumerate(n_components_list):
        pca = PCA(n_components=n_comp)
        X_transformed = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)

        reconstructed_digit = X_reconstructed[0].reshape(8, 8)
        axes[1, i].imshow(reconstructed_digit, cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title(
            f"{n_comp} components\n{pca.explained_variance_ratio_.sum():.2%} var"
        )

    plt.tight_layout()
    plt.savefig("assets/pca_image_compression.png")
    plt.close()

    # Explained variance plot
    pca = PCA()
    pca.fit(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cumsum) + 1), cumsum, "bo-")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("Explained Variance vs Number of Components")
    plt.grid(True)
    plt.savefig("assets/pca_explained_variance.png")
    plt.close()


# 2. t-SNE and UMAP Visualizations
def generate_tsne_umap_visualizations():
    # Create sample data
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Apply UMAP
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X)

    # Create visualization
    plt.figure(figsize=(15, 5))

    # Original data
    plt.subplot(131)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")
    plt.title("Original Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # t-SNE visualization
    plt.subplot(132)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="viridis")
    plt.title("t-SNE Visualization")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # UMAP visualization
    plt.subplot(133)
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap="viridis")
    plt.title("UMAP Visualization")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    plt.tight_layout()
    plt.savefig("assets/tsne_umap_comparison.png")
    plt.close()

    # Digits dataset visualization
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Apply UMAP
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X)

    # Create visualization
    plt.figure(figsize=(15, 5))

    # t-SNE visualization
    plt.subplot(121)
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10")
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of Digits")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # UMAP visualization
    plt.subplot(122)
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap="tab10")
    plt.colorbar(scatter)
    plt.title("UMAP Visualization of Digits")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    plt.tight_layout()
    plt.savefig("assets/tsne_umap_digits.png")
    plt.close()


# 3. Clustering Visualizations
def generate_clustering_visualizations():
    # Create sample data
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # K-Means
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")
    plt.title("Original Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.subplot(122)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap="viridis")
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        c="red",
        marker="x",
        s=200,
        linewidths=3,
        label="Centroids",
    )
    plt.title("K-Means Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

    plt.tight_layout()
    plt.savefig("assets/kmeans_example.png")
    plt.close()

    # Hierarchical Clustering
    model = AgglomerativeClustering(n_clusters=4)
    y_hc = model.fit_predict(X)

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")
    plt.title("Original Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.subplot(132)
    plt.scatter(X[:, 0], X[:, 1], c=y_hc, cmap="viridis")
    plt.title("Hierarchical Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.subplot(133)
    linkage_matrix = linkage(X, method="ward")
    dendrogram(linkage_matrix)
    plt.title("Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")

    plt.tight_layout()
    plt.savefig("assets/hierarchical_clustering.png")
    plt.close()

    # DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    y_dbscan = dbscan.fit_predict(X)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")
    plt.title("Original Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.subplot(122)
    plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap="viridis")
    plt.title("DBSCAN Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.tight_layout()
    plt.savefig("assets/dbscan_example.png")
    plt.close()

    # Elbow method
    inertias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), inertias, "bo-")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.grid(True)
    plt.savefig("assets/elbow_method.png")
    plt.close()


if __name__ == "__main__":
    generate_pca_visualizations()
    generate_tsne_umap_visualizations()
    generate_clustering_visualizations()
