# Neural Networks

Imagine your brain learning to recognize a cat - it processes information through layers of neurons, each detecting different features like edges, shapes, and patterns. Neural networks work similarly, learning hierarchical representations of data through interconnected layers. Let's explore this fascinating technology! 🧠

## Understanding Neural Networks 🎯

A neural network consists of:
1. Input Layer: Receives raw data
2. Hidden Layers: Process and transform data
3. Output Layer: Produces predictions

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Visualize network architecture
def plot_network_architecture():
    model = models.Sequential([
        layers.Dense(4, activation='relu', input_shape=(2,)),
        layers.Dense(4, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    tf.keras.utils.plot_model(
        model,
        show_shapes=True,
        show_layer_names=True,
        rankdir='LR'
    )

# Example will be shown in notebook environment
```

## Building Your First Neural Network 🏗️

### Simple Binary Classification
```python
import tensorflow as tf
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Create dataset
X, y = make_circles(n_samples=1000, noise=0.1, factor=0.3)
X = StandardScaler().fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build model
model = models.Sequential([
    layers.Dense(16, activation='relu', input_shape=(2,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
```

## Activation Functions 🔥

Different activation functions serve different purposes:

```python
def plot_activation_functions():
    x = np.linspace(-5, 5, 100)
    
    activations = {
        'ReLU': tf.nn.relu,
        'Sigmoid': tf.nn.sigmoid,
        'Tanh': tf.nn.tanh,
        'LeakyReLU': lambda x: tf.nn.leaky_relu(x, alpha=0.2)
    }
    
    plt.figure(figsize=(12, 4))
    for i, (name, func) in enumerate(activations.items(), 1):
        plt.subplot(1, 4, i)
        plt.plot(x, func(x))
        plt.title(name)
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_activation_functions()
```

## Real-World Example: Image Classification 🖼️

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and prepare MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize and reshape
X_train = X_train.reshape(-1, 28*28).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28*28).astype('float32') / 255.0

# Convert labels to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build model
model = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_acc:.3f}")

# Visualize predictions
def plot_predictions():
    n_samples = 5
    samples_idx = np.random.randint(0, len(X_test), n_samples)
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(samples_idx, 1):
        plt.subplot(1, n_samples, i)
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        pred = model.predict(X_test[idx:idx+1])
        plt.title(f'Pred: {pred.argmax()}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

plot_predictions()
```

## Advanced Concepts 🚀

### 1. Batch Normalization
```python
model = models.Sequential([
    layers.Dense(256, input_shape=(784,)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(10, activation='softmax')
])
```

### 2. Regularization
```python
from tensorflow.keras import regularizers

model = models.Sequential([
    layers.Dense(256, 
                kernel_regularizer=regularizers.l2(0.01),
                input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])
```

### 3. Custom Layers
```python
class ResidualBlock(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.dense1 = layers.Dense(units)
        self.dense2 = layers.Dense(units)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = layers.ReLU()(x)
        x = self.dense2(x)
        return layers.Add()([inputs, x])
```

## Best Practices 🌟

### 1. Data Preprocessing
```python
# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_clean = imputer.fit_transform(X_train)
```

### 2. Model Architecture
```python
def build_model(input_shape, n_classes):
    """Build model with best practices"""
    model = models.Sequential([
        layers.Dense(256, input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model
```

### 3. Training
```python
# Use learning rate scheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9
)

# Use early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train with best practices
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)
```

## Common Pitfalls and Solutions 🚧

1. **Vanishing/Exploding Gradients**
   - Use batch normalization
   - Try different activation functions
   - Initialize weights properly

2. **Overfitting**
   - Add dropout layers
   - Use regularization
   - Increase training data

3. **Poor Convergence**
   - Adjust learning rate
   - Try different optimizers
   - Check data preprocessing

## Next Steps

Congratulations! You now understand the basics of neural networks. Try the [assignment](./assignment.md) to practice these concepts!
