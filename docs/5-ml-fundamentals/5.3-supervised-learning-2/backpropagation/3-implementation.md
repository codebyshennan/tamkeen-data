---
reading_minutes: 25
objectives:
  - >-
    Implement `forward_pass` and `backward_pass` against a cached activations
    dict and verify shapes match each layer's weights.
  - >-
    Combine activation derivatives (sigmoid/ReLU/tanh) with the upstream `dz` to
    produce `dW`, `db`, and the next layer's `dz`.
  - >-
    Wire init/forward/backward/update into a small neural-network class and
    train it on a toy dataset to confirm the loss decreases.
---

# Implementing Backpropagation

**After this lesson:** you can explain Implementing Backpropagation and try the examples in your own notebook.

## Overview

Numerical vs symbolic gradients, modular layers, and debugging shape mistakes when implementing backward passes.

## Getting Started

Before we dive into the code, get clear on what we're building. We'll create a simple neural network that can learn from data. Think of it like teaching a computer to recognize patterns, similar to how you might teach a child to recognize different animals.

## Basic Implementation

Start with a basic implementation of backpropagation. This is like the core recipe for our neural network:

#### `backward_pass` skeleton

Output layer error (δ)

`dz = prediction - y` is the gradient of MSE loss w.r.t. the output pre-activation, the "how wrong were we" signal that flows backward through every layer.

Weight & bias gradients

`dW = dz · aᵀ`, outer product of upstream error and previous activations. `db = Σ dz` sums across the batch dimension. These gradients are what the optimizer uses to update each layer's parameters.

Propagate error backward

`dz = Wᵀ · dz × σ′(z)`, the chain rule in action. Transposing W "routes" the error back through the same weights that carried it forward. Multiplying by the activation derivative applies the local gradient at that layer.

## Activation Function Derivatives

These are like the rules for how the network should adjust its thinking:

#### Activation derivatives (elementwise)

Sigmoid derivative

Reuses the sigmoid output `sx` to compute `σ(x)·(1−σ(x))`, which is the local gradient passed back through any sigmoid activation during backprop.

ReLU derivative

`np.where` returns 1 where the pre-activation was positive and 0 elsewhere, the "dead neuron" property: no gradient flows for negative inputs.

Tanh derivative

`1 − tanh²(x)` is the exact derivative of tanh. Its output is always in (0, 1], so it avoids the vanishing-gradient extremes that plague sigmoid.

## Complete Implementation

Now, put it all together in a complete neural network class:

#### `NeuralNetwork` class (educational)

Weight initialisation

Weights are drawn from a zero-mean Gaussian scaled by 0.01 to keep activations in a well-behaved range at the start of training. Biases are initialised to zero, a safe default for fully-connected layers.

Forward pass

Each layer computes `z = W·a + b` then applies the activation. Both `z` and `a` are stashed in `cache` because the backward pass needs them to compute gradients.

Backward pass

Iterates layers in reverse, computing `dW`, `db`, then propagating `dz` further back via `Wᵀ · dz × σ′(z)`. The `if l > 0` guard prevents propagating past the input layer.

Parameter update (SGD)

Plain gradient descent: subtract `lr × gradient` from each weight and bias. Swap this method for Adam or RMSprop without touching the rest of the class.

Training loop

Runs forward → backward → update for each epoch, logging loss every 100 steps. The loss call re-reads the final activation from `cache` rather than doing a second forward pass.

Activation & loss helpers

Sigmoid activation and its derivative are defined here for self-containment. `compute_loss` uses MSE, switch to binary cross-entropy for classification tasks.

## Usage Example

Look at how to use our neural network:

Build the network

`[2, 3, 1]` means 2 input features, one hidden layer of 3 neurons, and 1 output neuron. Change these numbers to match your dataset's dimensions.

Synthetic data

Random data is used here for illustration. Note the shape convention: `(features, samples)`, columns are samples, rows are features, the opposite of the pandas/sklearn convention.

Train

Kicks off the forward → backward → update loop for 1000 epochs at `lr=0.01`. Loss is printed every 100 epochs so you can watch convergence.

Get predictions

Runs a forward pass and extracts the final layer's activations. The key is built dynamically from the number of weight matrices so it works regardless of network depth.

```
Epoch 0, Loss: 1.2398897398986224
Epoch 100, Loss: 0.9784079619497448
Epoch 200, Loss: 0.978407981338456
Epoch 300, Loss: 0.9784079813400142
Epoch 400, Loss: 0.9784079813400143
Epoch 500, Loss: 0.9784079813400143
Epoch 600, Loss: 0.9784079813400143
Epoch 700, Loss: 0.9784079813400143
Epoch 800, Loss: 0.9784079813400143
Epoch 900, Loss: 0.9784079813400143
```

## Visualizing the Training Process

Add some visualization to help understand what's happening:

Function signature

Accepts an already-constructed `NeuralNetwork` object so it works with any architecture. The `losses` list accumulates one scalar per epoch for plotting.

Training loop

Runs the full forward → loss → backward → update cycle every epoch and records the loss. Unlike `train()`, every epoch's loss is saved rather than only printing every 100 steps.

Loss curve plot

A simple matplotlib line plot of epoch vs loss. A steadily decreasing curve indicates healthy training; plateaus or oscillations suggest a learning-rate or architecture issue.

Driver code

Creates a fresh `[2, 3, 1]` network and random data, then calls the function, a minimal end-to-end demo you can run directly in a notebook cell.

<figure><img src="../../../../.gitbook/assets/3-implementation_fig_1 (1).png" alt="3-implementation"><figcaption><p>Figure 1: Training Loss Over Time</p></figcaption></figure>

## Best Practices

1. **Gradient Checking**
   * Verify your implementation by comparing numerical and analytical gradients
   * Use small networks for testing
   * Check each layer separately
2. **Learning Rate**
   * Start with a small learning rate (e.g., 0.01)
   * Use learning rate scheduling
   * Consider adaptive methods (Adam, RMSprop)
3. **Initialization**
   * Use proper weight initialization (Xavier, He)
   * Initialize biases to zero
   * Consider batch normalization
4. **Regularization**
   * Use L1/L2 regularization
   * Implement dropout
   * Apply early stopping
5. **Debugging**
   * Monitor loss during training
   * Check gradient magnitudes
   * Visualize activations and weights

## Common Mistakes to Avoid

1. **Forgetting to Normalize Data**
   * Always normalize your inputs
   * Check for outliers
   * Handle missing values
2. **Poor Learning Rate Choice**
   * Start small and increase if needed
   * Watch for oscillations
   * Use learning rate scheduling
3. **Ignoring Regularization**
   * Add dropout or L2 regularization
   * Monitor for overfitting
   * Use early stopping
4. **Matrix Dimension Mismatches**
   * Check shapes before operations
   * Use broadcasting carefully
   * Verify your matrix multiplications

## Gotchas

* **Shape convention mismatch between examples**: The `NeuralNetwork` class uses the `(features, samples)` convention (columns are samples), which is the _opposite_ of the pandas/sklearn convention. Passing an `(n_samples, n_features)` array directly into `network.train` silently transposes the learning problem and produces garbage gradients.
* **Recomputing loss from the forward pass cache instead of re-running forward**: In the `train` method, `compute_loss` reads `cache['a…']` from the _same_ forward pass that generated the current gradients. Calling `forward` again would be wasteful but also consistent; mixing cache reads with extra forward calls is a common source of subtle training-loop bugs.
* **Initializing weights with `randn * 0.01` for every architecture**, Scaling by 0.01 keeps initial activations small for shallow nets but starves gradients in deep nets. Use Xavier init for tanh/sigmoid layers and He init for ReLU; the scaffolding here always uses `* 0.01`, which will fail silently on deeper architectures.
* **The training loop doesn't shuffle data between epochs**: The `train` method runs gradient descent on the full dataset in a single step each epoch. Without mini-batching or shuffling, the gradient estimate is just full-batch GD; on real data this means the model sees the same ordering every time, which can introduce bias.
* **`activation_derivative` is hard-coded to sigmoid throughout**: The `NeuralNetwork` class uses sigmoid both as the activation and for its derivative. Swapping `activation` to ReLU without also updating `activation_derivative` produces incorrect gradients with no error, the network trains silently with the wrong math.
* **Gradient checking is described but not wired up**: The "Best Practices" section recommends gradient checking, but there is no utility function for it in this file. Without a numerical gradient check, a subtle sign error or index-off-by-one in the backward pass can go undetected for many training runs.

## Additional Resources

* [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - Free online book with interactive examples
* [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/) - Stanford's deep learning course
* [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - Visual explanations
* [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) - Andrew Ng's comprehensive course
