---
reading_minutes: 20
objectives:
  - >-
    Identify the right architecture for a task: CNN for image data, RNN/LSTM/GRU
    for sequences, Transformer for long-range dependencies.
  - >-
    Apply training refinements, Adam, learning-rate scheduling, batch
    normalization, dropout, and explain when each helps.
  - >-
    Use transfer learning by fine-tuning a pretrained backbone instead of
    training from scratch.
---

# Advanced Neural Network Techniques

**After this lesson:** you can explain Advanced Neural Network Techniques and try the examples in your own notebook.

## Overview

Architectures and training refinements (initialization, batch norm sketch, etc.) at intro-plus level.

## Welcome to Advanced Neural Networks

Ready to take your neural network skills to the next level? This guide will introduce you to advanced techniques that power state-of-the-art AI systems. Think of it like learning advanced cooking techniques after mastering the basics!

## Why Advanced Techniques Matter

Understanding advanced neural network techniques helps you:

* Solve more complex problems
* Build more efficient models
* Create cutting-edge AI applications
* Stay competitive in the field

## Advanced Architectures

### 1. Residual Networks (ResNet)

ResNet is like building a highway through your neural network - it allows information to flow more easily through deep networks by adding "shortcut" connections.

#### Real-World Analogy

Imagine you're learning to play a complex piece of music. Instead of starting from scratch each time, you can jump to different sections using bookmarks. ResNet works similarly, allowing the network to "jump" over layers when needed.

Imports

TensorFlow and the Keras `layers` module are imported; all building blocks (Conv2D, BatchNorm, Add, etc.) come from this single namespace.

Residual Block

Two Conv→BatchNorm passes process the input while the original tensor is kept as a shortcut; a 1×1 conv aligns channel dimensions when they differ before the skip connection is added.

Create ResNet

A stem (7×7 conv, pool) feeds four stacked residual blocks with progressively wider filters; global average pooling collapses spatial dims before the softmax classification head.

### 2. Attention Mechanism

Attention is like having a spotlight that helps the network focus on the most important parts of the input.

#### Real-World Analogy

When reading a book, you don't pay equal attention to every word. Some words are more important for understanding the story. Attention mechanisms work similarly, helping the network focus on relevant information.

Layer Setup

Two dense layers are created in `__init__`: `W` projects values to a learned space and `V` collapses that to a single scalar score per timestep.

Attention Forward Pass

Values are projected through `W` then scored with a tanh-activated `V`; softmax normalizes the scores into weights, which are multiplied by the original values and summed to form a context vector.

### 3. Transformer Architecture

Transformers are like having a team of experts who can communicate with each other to understand complex relationships in data.

#### Real-World Analogy

Imagine a group of experts in a meeting. Each expert can directly communicate with any other expert, and they all work together to solve a problem. Transformers work similarly, allowing different parts of the network to communicate directly.

Similarity and Scaling

Query-key dot products measure similarity; dividing by √d\_k prevents dot products from growing too large in high dimensions, which would saturate the softmax and kill gradients.

Mask and Weighted Output

An optional mask adds a large negative value to forbidden positions (e.g., future tokens) so their softmax weight becomes \~0; the final output is the weighted sum of value vectors.

## Advanced Training Techniques

### 1. Curriculum Learning

Curriculum learning is like teaching a child - start with simple concepts and gradually increase difficulty.

#### Real-World Example

When learning to play chess:

1. Start with basic piece movements
2. Learn simple strategies
3. Practice against easy opponents
4. Gradually face more challenging opponents

Class Setup

The generator stores the full dataset, labels, and a user-supplied `difficulty_fn` that scores each sample; `epoch` is tracked to gradually raise the difficulty ceiling.

Curriculum Batch

Each call computes a threshold that rises by 0.1 per epoch (capped at 1.0), filters to only samples below that difficulty, then randomly draws a batch, gradually exposing harder examples as training progresses.

### 2. Mixed Precision Training

Mixed precision training is like using different tools for different tasks - some operations are done with less precision to save memory and speed up training.

#### Real-World Analogy

When cooking, you might use precise measurements for baking (exact grams) but approximate measurements for cooking (handful of herbs). Mixed precision works similarly, using high precision where needed and lower precision where acceptable.

Set Global Policy

`Policy('mixed_float16')` instructs Keras to use float16 for most operations (faster on GPU) while keeping float32 for numerically sensitive steps; setting globally applies to all layers created after this call.

Model and Optimizer

Force the output layer to float32 for numerical stability in loss computation; wrap Adam with `LossScaleOptimizer` to automatically scale gradients and prevent float16 underflow during backpropagation.

## Common Mistakes to Avoid

1. **Using Advanced Techniques Unnecessarily**
   * Start with simple architectures
   * Only add complexity when needed
   * Monitor performance improvements
2. **Improper Implementation**
   * Test each component separately
   * Use appropriate initialization
   * Monitor training dynamics
3. **Memory Issues**
   * Use mixed precision when possible
   * Implement gradient checkpointing
   * Monitor GPU memory usage

## Practical Tips

1. **When to Use Advanced Architectures**
   * ResNet: Deep image classification
   * Attention: Sequence processing
   * Transformers: Language tasks
2. **Training Considerations**
   * Start with small learning rates
   * Use appropriate batch sizes
   * Monitor validation performance
3. **Performance Optimization**
   * Profile your code
   * Use appropriate hardware
   * Implement efficient data pipelines

## Gotchas

* **Residual block dimension mismatch crashes silently on some TF versions**: The `residual_block` function adjusts the shortcut with a 1×1 convolution when `shortcut.shape[-1] != filters`. This check only covers channel dimension mismatches. If you add `strides=2` to the main-path convolutions (for downsampling), the spatial size also mismatches, and the `Add` layer will fail at runtime with a cryptic shape error.
* **Setting `base_model.trainable = True` unfreezes all layers, not just the last few**: The fine-tuning pattern requires unfreezing only the tail of ResNet (e.g., `base_model.layers[-4:]`). Writing `base_model.trainable = True` unfreezes all 175+ ResNet layers, causing a dramatically larger parameter space that overfits quickly on small datasets.
* **`mixed_float16` silently keeps BatchNorm in float32**: Keras automatically keeps normalization layers in float32 even under `mixed_float16` policy, which is correct. But learners often check layer dtypes and assume the policy isn't working because they see float32 layers. This is intentional; the compute-intensive Dense and Conv layers run in float16.
* **`CurriculumDataGenerator` can produce empty batches**: If `threshold` is low (early training) and the `difficulty_fn` scores most samples above that threshold, `eligible_data` can be empty. `np.random.choice(0, size=batch_size)` will raise a `ValueError`. Always add a fallback (e.g., `if len(eligible_data) < batch_size: ...`) before production use.
* **Attention weights are summed, not concatenated**: In the `AttentionLayer`, the context vector is computed as `tf.reduce_sum(attention_weights * values, axis=1)`. Replacing `reduce_sum` with concatenation produces a tensor with the wrong shape for downstream layers and typically a large performance drop, since positional information is lost.
* **`scaled_dot_product_attention` mask convention uses large negative values, not zeros**: The mask adds `-1e9` to masked positions so softmax assigns near-zero weight to them. Adding 0 to masked positions (a common mistake) means those positions contribute equally to the output, breaking causal attention in decoder models.

## Next Steps

Ready to apply these techniques to real-world problems? Continue to [Applications](5-applications.md) to see how these advanced techniques are used in practice!
