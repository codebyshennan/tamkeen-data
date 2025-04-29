import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

# Set style
plt.style.use("seaborn-v0_8")
sns.set_theme()


def save_figure(name):
    """Save figure with high resolution"""
    plt.savefig(f"assets/{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_chain_rule():
    """Plot chain rule visualization"""
    x = np.linspace(-2, 2, 100)
    g = x**2  # g(x) = x^2
    f = np.sin(g)  # f(g(x)) = sin(x^2)

    plt.figure(figsize=(10, 6))
    plt.plot(x, g, label="g(x) = x^2", linewidth=2)
    plt.plot(x, f, label="f(g(x)) = sin(x^2)", linewidth=2)
    plt.legend(fontsize=12)
    plt.title("Chain Rule Visualization", fontsize=14)
    plt.grid(True)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    save_figure("chain_rule")


def plot_forward_pass():
    """Plot forward pass visualization"""
    plt.figure(figsize=(12, 8))

    # Create network layout
    G = nx.DiGraph()

    # Add nodes
    for i in range(3):  # Input layer
        G.add_node(f"input_{i}", pos=(0, i))
    for i in range(3):  # Hidden layer
        G.add_node(f"hidden_{i}", pos=(1, i))
    for i in range(2):  # Output layer
        G.add_node(f"output_{i}", pos=(2, i + 0.5))

    # Add edges
    for i in range(3):
        for j in range(3):
            G.add_edge(f"input_{i}", f"hidden_{j}")
    for i in range(3):
        for j in range(2):
            G.add_edge(f"hidden_{i}", f"output_{j}")

    # Draw network
    pos = nx.get_node_attributes(G, "pos")
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1000, alpha=0.7)
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20)

    # Add labels
    labels = {node: node.split("_")[0].title() for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=12)

    plt.title("Forward Pass Visualization", fontsize=14)
    plt.axis("off")
    save_figure("forward_pass")


def plot_activation_functions():
    """Plot activation functions and their derivatives"""
    x = np.linspace(-5, 5, 100)

    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-x))
    sigmoid_derivative = sigmoid * (1 - sigmoid)

    # ReLU
    relu = np.maximum(0, x)
    relu_derivative = np.where(x > 0, 1, 0)

    # Tanh
    tanh = np.tanh(x)
    tanh_derivative = 1 - np.tanh(x) ** 2

    plt.figure(figsize=(12, 8))

    # Plot functions
    plt.subplot(2, 1, 1)
    plt.plot(x, sigmoid, label="Sigmoid", linewidth=2)
    plt.plot(x, relu, label="ReLU", linewidth=2)
    plt.plot(x, tanh, label="Tanh", linewidth=2)
    plt.title("Activation Functions", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Plot derivatives
    plt.subplot(2, 1, 2)
    plt.plot(x, sigmoid_derivative, label="Sigmoid Derivative", linewidth=2)
    plt.plot(x, relu_derivative, label="ReLU Derivative", linewidth=2)
    plt.plot(x, tanh_derivative, label="Tanh Derivative", linewidth=2)
    plt.title("Activation Function Derivatives", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    save_figure("activation_functions")


def plot_loss_functions():
    """Plot loss functions"""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.linspace(0.01, 0.99, 100)

    # MSE
    mse = np.array([np.mean((y_true - p) ** 2) for p in y_pred])

    # BCE
    bce = np.array(
        [-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)) for p in y_pred]
    )

    plt.figure(figsize=(10, 6))
    plt.plot(y_pred, mse, label="MSE", linewidth=2)
    plt.plot(y_pred, bce, label="BCE", linewidth=2)
    plt.title("Loss Functions", fontsize=14)
    plt.xlabel("Prediction", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    save_figure("loss_functions")


def plot_training_process():
    """Plot training process visualization"""
    # Generate synthetic training data
    np.random.seed(42)
    X = np.random.randn(2, 1000)
    y = np.random.randn(1, 1000)

    # Simulate training process
    epochs = 1000
    learning_rate = 0.01
    losses = []

    # Initialize network
    weights = [np.random.randn(3, 2) * 0.01, np.random.randn(1, 3) * 0.01]
    biases = [np.zeros((3, 1)), np.zeros((1, 1))]

    for epoch in range(epochs):
        # Forward pass
        z1 = np.dot(weights[0], X) + biases[0]
        a1 = 1 / (1 + np.exp(-z1))
        z2 = np.dot(weights[1], a1) + biases[1]
        a2 = 1 / (1 + np.exp(-z2))

        # Compute loss
        loss = np.mean((y - a2) ** 2)
        losses.append(loss)

        # Backward pass (simplified)
        dz2 = a2 - y
        dw2 = np.dot(dz2, a1.T)
        db2 = np.sum(dz2, axis=1, keepdims=True)

        dz1 = np.dot(weights[1].T, dz2) * a1 * (1 - a1)
        dw1 = np.dot(dz1, X.T)
        db1 = np.sum(dz1, axis=1, keepdims=True)

        # Update weights
        weights[0] -= learning_rate * dw1
        biases[0] -= learning_rate * db1
        weights[1] -= learning_rate * dw2
        biases[1] -= learning_rate * db2

    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.title("Training Loss Over Time", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True)
    save_figure("training_process")


def plot_gradient_flow():
    """Plot gradient flow visualization"""
    plt.figure(figsize=(12, 8))

    # Create network layout
    G = nx.DiGraph()

    # Add nodes
    for i in range(3):  # Input layer
        G.add_node(f"input_{i}", pos=(0, i))
    for i in range(3):  # Hidden layer
        G.add_node(f"hidden_{i}", pos=(1, i))
    for i in range(2):  # Output layer
        G.add_node(f"output_{i}", pos=(2, i + 0.5))

    # Add edges with gradient flow
    for i in range(3):
        for j in range(3):
            G.add_edge(f"input_{i}", f"hidden_{j}", weight=np.random.rand())
    for i in range(3):
        for j in range(2):
            G.add_edge(f"hidden_{i}", f"output_{j}", weight=np.random.rand())

    # Draw network
    pos = nx.get_node_attributes(G, "pos")
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1000, alpha=0.7)
    nx.draw_networkx_edges(
        G, pos, edge_color="gray", width=edge_weights, arrows=True, arrowsize=20
    )

    # Add labels
    labels = {node: node.split("_")[0].title() for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=12)

    plt.title("Gradient Flow Visualization", fontsize=14)
    plt.axis("off")
    save_figure("gradient_flow")


def plot_vanishing_gradients():
    """Plot vanishing gradients visualization"""
    x = np.linspace(-5, 5, 100)
    sigmoid = 1 / (1 + np.exp(-x))
    sigmoid_derivative = sigmoid * (1 - sigmoid)

    plt.figure(figsize=(10, 6))
    plt.plot(x, sigmoid_derivative, label="Sigmoid Derivative", linewidth=2)
    plt.title("Vanishing Gradients Problem", fontsize=14)
    plt.xlabel("Input", fontsize=12)
    plt.ylabel("Gradient", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Add annotations
    plt.annotate(
        "Small gradients\nin these regions",
        xy=(-4, 0.1),
        xytext=(-3, 0.3),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )
    plt.annotate(
        "Small gradients\nin these regions",
        xy=(4, 0.1),
        xytext=(3, 0.3),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )

    save_figure("vanishing_gradients")


def plot_exploding_gradients():
    """Plot exploding gradients visualization"""
    x = np.linspace(-5, 5, 100)
    tanh = np.tanh(x)
    tanh_derivative = 1 - np.tanh(x) ** 2

    plt.figure(figsize=(10, 6))
    plt.plot(x, tanh_derivative, label="Tanh Derivative", linewidth=2)
    plt.title("Exploding Gradients Problem", fontsize=14)
    plt.xlabel("Input", fontsize=12)
    plt.ylabel("Gradient", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Add annotations
    plt.annotate(
        "Large gradients\nin these regions",
        xy=(-2, 0.8),
        xytext=(-3, 0.5),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )
    plt.annotate(
        "Large gradients\nin these regions",
        xy=(2, 0.8),
        xytext=(3, 0.5),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )

    save_figure("exploding_gradients")


def main():
    """Generate all graphics"""
    print("Generating graphics...")

    # Create graphics
    plot_chain_rule()
    plot_forward_pass()
    plot_activation_functions()
    plot_loss_functions()
    plot_training_process()
    plot_gradient_flow()
    plot_vanishing_gradients()
    plot_exploding_gradients()

    print("Graphics generated successfully!")


if __name__ == "__main__":
    main()
