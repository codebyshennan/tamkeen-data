# Implementing Neural Networks 💻

Let's explore how to implement Neural Networks using popular frameworks like TensorFlow and PyTorch!

## Basic Implementation with TensorFlow 🚀

### Simple Classification Example
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create sample dataset
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
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
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True
        )
    ]
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f"Test accuracy: {test_acc:.3f}")
```

## PyTorch Implementation 🔥

### Custom Neural Network
```python
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test)

# Create model
model = NeuralNet(
    input_size=20,
    hidden_size=64,
    num_classes=1
)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Training loop
num_epochs = 50
batch_size = 32

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor.view(-1, 1))
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test accuracy
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = (outputs > 0.5).float()
    accuracy = (predicted == y_test_tensor.view(-1, 1)).float().mean()
    print(f'Test Accuracy: {accuracy:.3f}')
```

## Real-World Example: Image Classification 🖼️

### CNN with TensorFlow
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create CNN model
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', 
                     input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load and prepare data
train_generator = train_datagen.flow_from_directory(
    'train_dir',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    'validation_dir',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Create and compile model
model = create_cnn_model(
    input_shape=(224, 224, 3),
    num_classes=len(train_generator.class_indices)
)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=3
        )
    ]
)
```

## Transfer Learning Example 🔄

### Using Pre-trained ResNet
```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Create base model
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Create new model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32
)

# Fine-tuning
base_model.trainable = True

# Freeze early layers
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32
)
```

## Best Practices 🌟

### 1. Model Architecture
```python
def create_model(input_shape, num_classes):
    """Create model with best practices"""
    model = models.Sequential([
        # Input normalization
        layers.BatchNormalization(input_shape=input_shape),
        
        # First block
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.3),
        
        # Second block
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
```

### 2. Training Configuration
```python
def get_training_config():
    """Get training configuration"""
    return {
        'optimizer': tf.keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0
        ),
        'loss': 'categorical_crossentropy',
        'metrics': [
            'accuracy',
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ],
        'callbacks': [
            tf.keras.callbacks.EarlyStopping(
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=3,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                save_best_only=True
            )
        ]
    }
```

### 3. Data Pipeline
```python
def create_data_pipeline(data_dir, batch_size=32):
    """Create efficient data pipeline"""
    return tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(224, 224),
        batch_size=batch_size
    ).map(lambda x, y: (
        tf.keras.applications.resnet50.preprocess_input(x), y
    )).prefetch(tf.data.AUTOTUNE)
```

## Common Pitfalls and Solutions 🚧

1. **Vanishing/Exploding Gradients**
   ```python
   # Use gradient clipping
   optimizer = tf.keras.optimizers.Adam(
       clipnorm=1.0,
       clipvalue=0.5
   )
   ```

2. **Overfitting**
   ```python
   # Add regularization
   layers.Dense(
       256,
       kernel_regularizer=tf.keras.regularizers.l2(0.01),
       activity_regularizer=tf.keras.regularizers.l1(0.01)
   )
   ```

3. **Memory Issues**
   ```python
   # Use generator for large datasets
   train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
       preprocessing_function=preprocess_input
   ).flow_from_directory(
       'train_dir',
       target_size=(224, 224),
       batch_size=32
   )
   ```

## Next Steps 🚀

Ready to explore advanced techniques? Continue to [Advanced Topics](4-advanced.md)!
