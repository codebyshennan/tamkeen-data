---
reading_minutes: 20
objectives:
  - "Identify the right architecture for a task: CNN for image data, RNN/LSTM/GRU for sequences, Transformer for long-range dependencies."
  - "Apply training refinements — Adam, learning-rate scheduling, batch normalization, dropout — and explain when each helps."
  - "Use transfer learning by fine-tuning a pretrained backbone instead of training from scratch."
---

# Advanced Neural Network Techniques

**After this lesson:** you can explain the core ideas in “Advanced Neural Network Techniques” and reproduce the examples here in your own notebook or environment.

## Overview

Architectures and training refinements (initialization, batch norm sketch, etc.) at intro-plus level.

## Helpful video

Crash Course AI: supervised learning framing (~15 min).

<iframe width="560" height="315" src="https://www.youtube.com/embed/4qVRBYAdLAo" title="Supervised Learning: Crash Course AI" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Welcome to Advanced Neural Networks

Ready to take your neural network skills to the next level? This guide will introduce you to advanced techniques that power state-of-the-art AI systems. Think of it like learning advanced cooking techniques after mastering the basics!

## Why Advanced Techniques Matter

Understanding advanced neural network techniques helps you:

- Solve more complex problems
- Build more efficient models
- Create cutting-edge AI applications
- Stay competitive in the field

## Advanced Architectures

{% include mermaid-diagram.html src="5-ml-fundamentals/5.3-supervised-learning-2/neural-networks/diagrams/4-advanced-1.mmd" %}

### 1. Residual Networks (ResNet)

ResNet is like building a highway through your neural network - it allows information to flow more easily through deep networks by adding "shortcut" connections.

#### Real-World Analogy

Imagine you're learning to play a complex piece of music. Instead of starting from scratch each time, you can jump to different sections using bookmarks. ResNet works similarly, allowing the network to "jump" over layers when needed.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import tensorflow as tf
from tensorflow.keras import layers

def residual_block(x, filters, kernel_size=3):
    """Create a residual block with shortcut connection"""
    # Save the input for the shortcut
    shortcut = x

    # First convolution
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second convolution
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Add shortcut if dimensions don't match
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1)(shortcut)

    # Add shortcut to output
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x

# Create a simple ResNet for image classification
def create_resnet(input_shape, num_classes):
    """Create a ResNet model for image classification"""
    inputs = layers.Input(shape=input_shape)

    # Initial processing
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Stack of residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 128, strides=2)
    x = residual_block(x, 256, strides=2)
    x = residual_block(x, 512, strides=2)

    # Final processing
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-2" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports</span>
    </div>
    <div class="code-callout__body">
      <p>TensorFlow and the Keras <code>layers</code> module are imported; all building blocks (Conv2D, BatchNorm, Add, etc.) come from this single namespace.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="4-26" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Residual Block</span>
    </div>
    <div class="code-callout__body">
      <p>Two Conv→BatchNorm passes process the input while the original tensor is kept as a shortcut; a 1×1 conv aligns channel dimensions when they differ before the skip connection is added.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="28-50" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Create ResNet</span>
    </div>
    <div class="code-callout__body">
      <p>A stem (7×7 conv, pool) feeds four stacked residual blocks with progressively wider filters; global average pooling collapses spatial dims before the softmax classification head.</p>
    </div>
  </div>
</aside>
</div>

### 2. Attention Mechanism

Attention is like having a spotlight that helps the network focus on the most important parts of the input.

#### Real-World Analogy

When reading a book, you don't pay equal attention to every word. Some words are more important for understanding the story. Attention mechanisms work similarly, helping the network focus on relevant information.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class AttentionLayer(layers.Layer):
    """Custom attention layer for focusing on important features"""
    def __init__(self, units):
        super().__init__()
        self.W = layers.Dense(units)  # For processing the input
        self.V = layers.Dense(1)      # For computing attention scores

    def call(self, query, values):
        # Process the values
        processed_values = self.W(values)

        # Compute attention scores
        attention_scores = self.V(tf.nn.tanh(processed_values))

        # Convert scores to weights
        attention_weights = tf.nn.softmax(attention_scores, axis=1)

        # Apply weights to values
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-6" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Layer Setup</span>
    </div>
    <div class="code-callout__body">
      <p>Two dense layers are created in <code>__init__</code>: <code>W</code> projects values to a learned space and <code>V</code> collapses that to a single scalar score per timestep.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="8-22" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Attention Forward Pass</span>
    </div>
    <div class="code-callout__body">
      <p>Values are projected through <code>W</code> then scored with a tanh-activated <code>V</code>; softmax normalizes the scores into weights, which are multiplied by the original values and summed to form a context vector.</p>
    </div>
  </div>
</aside>
</div>

### 3. Transformer Architecture

Transformers are like having a team of experts who can communicate with each other to understand complex relationships in data.

#### Real-World Analogy

Imagine a group of experts in a meeting. Each expert can directly communicate with any other expert, and they all work together to solve a problem. Transformers work similarly, allowing different parts of the network to communicate directly.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculate attention weights using scaled dot product"""
    # Compute similarity between query and key
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # Scale by square root of dimension
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Apply mask if provided
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Convert to probabilities
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1
    )

    # Apply weights to values
    output = tf.matmul(attention_weights, v)
    return output, attention_weights
{% endhighlight %}

<figure>
<img src="assets/4-advanced_fig_1.png" alt="4-advanced" />
<figcaption>Figure 1: Generated visualization</figcaption>
</figure>


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-9" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Similarity and Scaling</span>
    </div>
    <div class="code-callout__body">
      <p>Query–key dot products measure similarity; dividing by √d_k prevents dot products from growing too large in high dimensions, which would saturate the softmax and kill gradients.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="11-21" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Mask and Weighted Output</span>
    </div>
    <div class="code-callout__body">
      <p>An optional mask adds a large negative value to forbidden positions (e.g., future tokens) so their softmax weight becomes ~0; the final output is the weighted sum of value vectors.</p>
    </div>
  </div>
</aside>
</div>

## Advanced Training Techniques

### 1. Curriculum Learning

Curriculum learning is like teaching a child - start with simple concepts and gradually increase difficulty.

#### Real-World Example

When learning to play chess:

1. Start with basic piece movements
2. Learn simple strategies
3. Practice against easy opponents
4. Gradually face more challenging opponents

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class CurriculumDataGenerator:
    """Generate data with increasing difficulty"""
    def __init__(self, data, labels, difficulty_fn):
        self.data = data
        self.labels = labels
        self.difficulty_fn = difficulty_fn
        self.epoch = 0

    def get_batch(self, batch_size):
        # Calculate current difficulty threshold
        threshold = min(1.0, 0.2 + 0.1 * self.epoch)

        # Get sample difficulties
        difficulties = self.difficulty_fn(self.data)

        # Select samples below threshold
        mask = difficulties <= threshold
        eligible_data = self.data[mask]
        eligible_labels = self.labels[mask]

        # Sample batch
        indices = np.random.choice(
            len(eligible_data),
            size=batch_size
        )

        return eligible_data[indices], eligible_labels[indices]
{% endhighlight %}

<figure>
<img src="assets/4-advanced_fig_1.png" alt="4-advanced" />
<figcaption>Figure 1: Generated visualization</figcaption>
</figure>


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Class Setup</span>
    </div>
    <div class="code-callout__body">
      <p>The generator stores the full dataset, labels, and a user-supplied <code>difficulty_fn</code> that scores each sample; <code>epoch</code> is tracked to gradually raise the difficulty ceiling.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="10-27" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Curriculum Batch</span>
    </div>
    <div class="code-callout__body">
      <p>Each call computes a threshold that rises by 0.1 per epoch (capped at 1.0), filters to only samples below that difficulty, then randomly draws a batch—gradually exposing harder examples as training progresses.</p>
    </div>
  </div>
</aside>
</div>

### 2. Mixed Precision Training

Mixed precision training is like using different tools for different tasks - some operations are done with less precision to save memory and speed up training.

#### Real-World Analogy

When cooking, you might use precise measurements for baking (exact grams) but approximate measurements for cooking (handful of herbs). Mixed precision works similarly, using high precision where needed and lower precision where acceptable.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
# Enable mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Create model with mixed precision
def create_mixed_precision_model():
    model = create_model()  # Your model architecture

    # Ensure last layer uses float32 for stability
    model.outputs[0].dtype = 'float32'

    # Use mixed precision optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
        optimizer
    )

    return model, optimizer
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-3" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Set Global Policy</span>
    </div>
    <div class="code-callout__body">
      <p><code>Policy('mixed_float16')</code> instructs Keras to use float16 for most operations (faster on GPU) while keeping float32 for numerically sensitive steps; setting globally applies to all layers created after this call.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="5-19" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Model and Optimizer</span>
    </div>
    <div class="code-callout__body">
      <p>Force the output layer to float32 for numerical stability in loss computation; wrap Adam with <code>LossScaleOptimizer</code> to automatically scale gradients and prevent float16 underflow during backpropagation.</p>
    </div>
  </div>
</aside>
</div>

## Common Mistakes to Avoid

1. **Using Advanced Techniques Unnecessarily**
   - Start with simple architectures
   - Only add complexity when needed
   - Monitor performance improvements

2. **Improper Implementation**
   - Test each component separately
   - Use appropriate initialization
   - Monitor training dynamics

3. **Memory Issues**
   - Use mixed precision when possible
   - Implement gradient checkpointing
   - Monitor GPU memory usage

## Practical Tips

1. **When to Use Advanced Architectures**
   - ResNet: Deep image classification
   - Attention: Sequence processing
   - Transformers: Language tasks

2. **Training Considerations**
   - Start with small learning rates
   - Use appropriate batch sizes
   - Monitor validation performance

3. **Performance Optimization**
   - Profile your code
   - Use appropriate hardware
   - Implement efficient data pipelines

## Gotchas

- **Residual block dimension mismatch crashes silently on some TF versions** — The `residual_block` function adjusts the shortcut with a 1×1 convolution when `shortcut.shape[-1] != filters`. This check only covers channel dimension mismatches. If you add `strides=2` to the main-path convolutions (for downsampling), the spatial size also mismatches, and the `Add` layer will fail at runtime with a cryptic shape error.
- **Setting `base_model.trainable = True` unfreezes all layers, not just the last few** — The fine-tuning pattern requires unfreezing only the tail of ResNet (e.g., `base_model.layers[-4:]`). Writing `base_model.trainable = True` unfreezes all 175+ ResNet layers, causing a dramatically larger parameter space that overfits quickly on small datasets.
- **`mixed_float16` silently keeps BatchNorm in float32** — Keras automatically keeps normalization layers in float32 even under `mixed_float16` policy, which is correct. But learners often check layer dtypes and assume the policy isn't working because they see float32 layers. This is intentional; the compute-intensive Dense and Conv layers run in float16.
- **`CurriculumDataGenerator` can produce empty batches** — If `threshold` is low (early training) and the `difficulty_fn` scores most samples above that threshold, `eligible_data` can be empty. `np.random.choice(0, size=batch_size)` will raise a `ValueError`. Always add a fallback (e.g., `if len(eligible_data) < batch_size: ...`) before production use.
- **Attention weights are summed, not concatenated** — In the `AttentionLayer`, the context vector is computed as `tf.reduce_sum(attention_weights * values, axis=1)`. Replacing `reduce_sum` with concatenation produces a tensor with the wrong shape for downstream layers and typically a large performance drop, since positional information is lost.
- **`scaled_dot_product_attention` mask convention uses large negative values, not zeros** — The mask adds `-1e9` to masked positions so softmax assigns near-zero weight to them. Adding 0 to masked positions (a common mistake) means those positions contribute equally to the output, breaking causal attention in decoder models.

## Next Steps

Ready to apply these techniques to real-world problems? Continue to [Applications](5-applications.md) to see how these advanced techniques are used in practice!
