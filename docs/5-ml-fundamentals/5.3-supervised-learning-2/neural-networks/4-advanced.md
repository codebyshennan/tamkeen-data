# Advanced Neural Network Techniques 🚀

Let's explore advanced concepts and architectures that take Neural Networks to the next level!

## Advanced Architectures 🏗️

### 1. Residual Networks (ResNet)
```python
import tensorflow as tf
from tensorflow.keras import layers

def residual_block(x, filters, kernel_size=3):
    """Create a residual block"""
    shortcut = x
    
    # First convolution
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Second convolution
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Add shortcut
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1)(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    
    return x

def create_resnet(input_shape, num_classes):
    """Create ResNet model"""
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 128, strides=2)
    x = residual_block(x, 256, strides=2)
    x = residual_block(x, 512, strides=2)
    
    # Output
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)
```

### 2. Attention Mechanism
```python
class AttentionLayer(layers.Layer):
    """Custom attention layer"""
    def __init__(self, units):
        super().__init__()
        self.W = layers.Dense(units)
        self.V = layers.Dense(1)
    
    def call(self, query, values):
        # Query shape: [batch_size, query_len, units]
        # Values shape: [batch_size, value_len, units]
        
        # Score shape: [batch_size, value_len, 1]
        score = self.V(tf.nn.tanh(self.W(values)))
        
        # Attention weights shape: [batch_size, value_len, 1]
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Context vector shape: [batch_size, units]
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights
```

### 3. Transformer Architecture
```python
def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculate attention weights"""
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    # Scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    # Add mask if provided
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    # Softmax
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1
    )
    
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(layers.Layer):
    """Multi-head attention layer"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )
        
        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )
        
        output = self.dense(concat_attention)
        return output, attention_weights
```

## Advanced Training Techniques 🎓

### 1. Curriculum Learning
```python
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
```

### 2. Mixed Precision Training
```python
# Enable mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Create model with mixed precision
def create_mixed_precision_model():
    model = create_model()  # Your model architecture
    
    # Ensure last layer uses float32
    model.outputs[0].dtype = 'float32'
    
    # Use mixed precision optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
        optimizer
    )
    
    return model, optimizer
```

### 3. Progressive Resizing
```python
def train_with_progressive_resizing(
    model, train_data, initial_size=64, final_size=224,
    size_increment=32, epochs_per_size=5):
    """Train model with progressively larger images"""
    current_size = initial_size
    
    while current_size <= final_size:
        print(f"Training with size: {current_size}")
        
        # Resize dataset
        resized_data = tf.image.resize(
            train_data,
            (current_size, current_size)
        )
        
        # Train for some epochs
        model.fit(
            resized_data,
            epochs=epochs_per_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=2)
            ]
        )
        
        current_size += size_increment
```

## Advanced Loss Functions 📉

### 1. Focal Loss
```python
class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss for imbalanced datasets"""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        # Calculate cross entropy
        ce = tf.keras.losses.binary_crossentropy(
            y_true,
            y_pred
        )
        
        # Calculate focal term
        p_t = (y_true * y_pred) + (
            (1 - y_true) * (1 - y_pred)
        )
        focal_term = tf.pow(1 - p_t, self.gamma)
        
        # Calculate alpha term
        alpha_term = y_true * self.alpha + (
            1 - y_true
        ) * (1 - self.alpha)
        
        return tf.reduce_mean(
            alpha_term * focal_term * ce
        )
```

### 2. Contrastive Loss
```python
def contrastive_loss(y_true, embeddings, margin=1.0):
    """Contrastive loss for siamese networks"""
    # Calculate pairwise distances
    distances = tf.reduce_sum(
        tf.square(embeddings[0] - embeddings[1]),
        axis=1
    )
    
    # Calculate loss
    similar_loss = y_true * distances
    dissimilar_loss = (1 - y_true) * tf.maximum(
        0.0,
        margin - distances
    )
    
    return tf.reduce_mean(
        similar_loss + dissimilar_loss
    )
```

## Advanced Regularization 🎛️

### 1. Stochastic Depth
```python
class StochasticDepth(layers.Layer):
    """Stochastic Depth layer"""
    def __init__(self, survival_probability=0.8):
        super().__init__()
        self.survival_probability = survival_probability
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        # Create random tensor
        batch_size = tf.shape(inputs)[0]
        random_tensor = self.survival_probability
        random_tensor += tf.random.uniform(
            [batch_size, 1, 1, 1],
            dtype=inputs.dtype
        )
        binary_tensor = tf.floor(random_tensor)
        
        return inputs * binary_tensor / self.survival_probability
```

### 2. Mixup
```python
def mixup_data(x, y, alpha=0.2):
    """Perform mixup on the input data"""
    # Generate mixup weights
    weights = np.random.beta(alpha, alpha, size=len(x))
    x_weights = weights.reshape(len(x), 1, 1, 1)
    y_weights = weights.reshape(len(x), 1)
    
    # Create shuffled indices
    index = np.random.permutation(len(x))
    
    # Create mixup samples
    x_mixup = (
        x * x_weights +
        x[index] * (1 - x_weights)
    )
    y_mixup = (
        y * y_weights +
        y[index] * (1 - y_weights)
    )
    
    return x_mixup, y_mixup
```

## Model Interpretation 🔍

### 1. Grad-CAM
```python
def grad_cam(model, image, layer_name):
    """Generate Grad-CAM heatmap"""
    # Get gradient model
    grad_model = tf.keras.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    
    # Calculate gradients
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    
    # Extract gradients
    grads = tape.gradient(loss, conv_output)
    
    # Calculate guided gradients
    guided_grads = tf.cast(conv_output > 0, 'float32') * grads
    
    # Calculate weights
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    
    # Generate heatmap
    cam = tf.reduce_sum(
        tf.multiply(weights, conv_output),
        axis=-1
    )
    
    return cam.numpy()
```

### 2. Integrated Gradients
```python
def integrated_gradients(model, image, baseline=None, steps=50):
    """Calculate integrated gradients"""
    if baseline is None:
        baseline = tf.zeros_like(image)
    
    # Generate alphas
    alphas = tf.linspace(0.0, 1.0, steps)
    
    # Generate interpolated images
    interpolated = [
        baseline + alpha * (image - baseline)
        for alpha in alphas
    ]
    
    # Calculate gradients
    grads = []
    for interp in interpolated:
        with tf.GradientTape() as tape:
            tape.watch(interp)
            output = model(interp)
        grad = tape.gradient(output, interp)
        grads.append(grad)
    
    # Calculate integral
    grads = tf.convert_to_tensor(grads, dtype=tf.float32)
    avg_grads = tf.reduce_mean(grads, axis=0)
    integrated_grads = (image - baseline) * avg_grads
    
    return integrated_grads
```

## Next Steps 🎯

Ready to see Neural Networks in action? Continue to [Applications](5-applications.md) to explore real-world use cases!
