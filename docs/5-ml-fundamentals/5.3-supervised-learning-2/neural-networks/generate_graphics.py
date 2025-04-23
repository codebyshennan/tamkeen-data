import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle, Rectangle

# Create assets directory if it doesn't exist
assets_dir = "assets"
os.makedirs(assets_dir, exist_ok=True)

# Set style
plt.style.use("default")
sns.set_palette("husl")


def save_fig(name):
    """Save figure with high resolution"""
    plt.savefig(f"{assets_dir}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


# 1. Introduction Graphics
def create_neural_network_diagram():
    """Create a simple neural network diagram"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw layers
    layer_sizes = [4, 5, 3, 2]
    layer_positions = np.linspace(0, 1, len(layer_sizes))

    for i, (pos, size) in enumerate(zip(layer_positions, layer_sizes)):
        # Draw neurons
        neuron_positions = np.linspace(0, 1, size)
        for j, neuron_pos in enumerate(neuron_positions):
            circle = Circle((pos, neuron_pos), 0.03, color="skyblue", alpha=0.8)
            ax.add_patch(circle)

            # Add connections
            if i < len(layer_sizes) - 1:
                next_pos = layer_positions[i + 1]
                next_size = layer_sizes[i + 1]
                next_positions = np.linspace(0, 1, next_size)
                for next_pos_y in next_positions:
                    ax.plot(
                        [pos, next_pos],
                        [neuron_pos, next_pos_y],
                        "gray",
                        alpha=0.3,
                        linewidth=0.5,
                    )

    # Add labels
    layer_names = ["Input", "Hidden", "Hidden", "Output"]
    for i, (pos, name) in enumerate(zip(layer_positions, layer_names)):
        ax.text(pos, -0.1, name, ha="center", fontsize=12)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 1.1)
    ax.axis("off")
    save_fig("neural_network_diagram")


# 2. Math Foundation Graphics
def create_activation_functions():
    """Plot common activation functions"""
    x = np.linspace(-5, 5, 100)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-x))
    axes[0].plot(x, sigmoid)
    axes[0].set_title("Sigmoid")
    axes[0].grid(True)

    # ReLU
    relu = np.maximum(0, x)
    axes[1].plot(x, relu)
    axes[1].set_title("ReLU")
    axes[1].grid(True)

    # Tanh
    tanh = np.tanh(x)
    axes[2].plot(x, tanh)
    axes[2].set_title("Tanh")
    axes[2].grid(True)

    plt.tight_layout()
    save_fig("activation_functions")


def create_loss_functions():
    """Plot common loss functions"""
    x = np.linspace(0, 1, 100)
    y_true = 0.5

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MSE
    mse = (x - y_true) ** 2
    axes[0].plot(x, mse)
    axes[0].set_title("Mean Squared Error")
    axes[0].grid(True)

    # Binary Cross-Entropy
    bce = -(y_true * np.log(x) + (1 - y_true) * np.log(1 - x))
    axes[1].plot(x, bce)
    axes[1].set_title("Binary Cross-Entropy")
    axes[1].grid(True)

    plt.tight_layout()
    save_fig("loss_functions")


# 3. Implementation Graphics
def create_training_curves():
    """Create example training curves"""
    epochs = np.arange(50)
    train_loss = np.exp(-epochs / 10) + np.random.normal(0, 0.02, 50)
    val_loss = np.exp(-epochs / 15) + np.random.normal(0, 0.02, 50)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    save_fig("training_curves")


# 4. Advanced Topics Graphics
def create_resnet_block():
    """Create ResNet block diagram"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw main path
    ax.plot([0, 1], [0.5, 0.5], "b-", linewidth=2)
    ax.plot([1, 2], [0.5, 0.5], "b-", linewidth=2)

    # Draw shortcut
    ax.plot([0, 2], [0.5, 0.5], "r--", linewidth=2)

    # Add blocks
    rect1 = Rectangle((0.2, 0.3), 0.6, 0.4, facecolor="lightblue", alpha=0.5)
    rect2 = Rectangle((1.2, 0.3), 0.6, 0.4, facecolor="lightblue", alpha=0.5)
    ax.add_patch(rect1)
    ax.add_patch(rect2)

    # Add labels
    ax.text(0.5, 0.5, "Conv + BN + ReLU", ha="center", va="center")
    ax.text(1.5, 0.5, "Conv + BN", ha="center", va="center")
    ax.text(1, 0.7, "Shortcut Connection", ha="center", va="center", color="red")

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.axis("off")
    save_fig("resnet_block")


# 5. Applications Graphics
def create_cv_nlp_ts_diagrams():
    """Create diagrams for CV, NLP, and Time Series applications"""
    # Computer Vision
    plt.figure(figsize=(8, 6))
    plt.imshow(np.random.rand(28, 28), cmap="gray")
    plt.title("Image Classification Example")
    plt.axis("off")
    save_fig("cv_example")

    # NLP
    plt.figure(figsize=(10, 4))
    text = "This product is amazing! I love it!"
    sentiment = 0.8
    plt.barh(["Positive", "Negative", "Neutral"], [sentiment, 0.1, 0.1])
    plt.title("Sentiment Analysis Example")
    plt.xlim(0, 1)
    save_fig("nlp_example")

    # Time Series
    plt.figure(figsize=(10, 4))
    t = np.arange(100)
    data = np.sin(t / 10) + np.random.normal(0, 0.1, 100)
    plt.plot(t, data)
    plt.title("Time Series Prediction Example")
    plt.xlabel("Time")
    plt.ylabel("Value")
    save_fig("ts_example")


# Generate all graphics
create_neural_network_diagram()
create_activation_functions()
create_loss_functions()
create_training_curves()
create_resnet_block()
create_cv_nlp_ts_diagrams()
