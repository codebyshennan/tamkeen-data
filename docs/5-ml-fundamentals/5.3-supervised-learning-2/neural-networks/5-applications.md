# Real-World Applications of Neural Networks 🌍

Let's explore how Neural Networks are used to solve real-world problems across different industries!

## 1. Computer Vision Applications 👁️

### Image Classification
```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_image_classifier(num_classes):
    """Create transfer learning image classifier"""
    # Load pre-trained model
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add custom layers
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
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

# Create data generators
train_generator = train_datagen.flow_from_directory(
    'train_dir',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Train model
model = create_image_classifier(num_classes=10)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=20,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3)
    ]
)
```

### Object Detection
```python
from tensorflow.keras.applications import YOLO

def create_object_detector():
    """Create YOLO object detector"""
    model = YOLO('yolov8n.pt')  # Load pre-trained model
    
    # Custom training settings
    model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        save=True
    )
    
    return model

def detect_objects(model, image):
    """Detect objects in image"""
    results = model(image)
    
    # Process results
    boxes = results[0].boxes
    for box in boxes:
        # Get coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        class_id = box.cls[0]
        
        print(f"Object: {class_id}, Confidence: {confidence:.2f}")
```

## 2. Natural Language Processing 📝

### Text Classification
```python
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

def create_text_classifier(num_labels):
    """Create BERT text classifier"""
    # Load pre-trained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels
    )
    
    return model, tokenizer

def prepare_text_data(texts, tokenizer, max_length=128):
    """Prepare text data for BERT"""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='tf'
    )

# Example usage
texts = [
    "This product is amazing!",
    "Terrible customer service",
    "Neutral experience overall"
]
labels = [1, 0, 2]  # Positive, Negative, Neutral

model, tokenizer = create_text_classifier(num_labels=3)
inputs = prepare_text_data(texts, tokenizer)

# Train model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(
    inputs,
    labels,
    epochs=3,
    batch_size=32
)
```

### Machine Translation
```python
from transformers import MarianMTModel, MarianTokenizer

def create_translator(src_lang='en', tgt_lang='fr'):
    """Create machine translation model"""
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    return model, tokenizer

def translate_text(text, model, tokenizer):
    """Translate text"""
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', padding=True)
    
    # Generate translation
    outputs = model.generate(**inputs)
    
    # Decode
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return translation
```

## 3. Time Series Applications 📈

### Stock Price Prediction
```python
import numpy as np
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

def create_stock_predictor(sequence_length):
    """Create LSTM model for stock prediction"""
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_sequences(data, sequence_length):
    """Prepare sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

# Example usage
import yfinance as yf

# Get stock data
stock = yf.Ticker('AAPL')
data = stock.history(period='1y')['Close'].values
data = (data - data.mean()) / data.std()  # Normalize

# Prepare sequences
sequence_length = 10
X, y = prepare_sequences(data, sequence_length)

# Create and train model
model = create_stock_predictor(sequence_length)
model.fit(X, y, epochs=50, batch_size=32)
```

### Anomaly Detection
```python
class TimeSeriesAnomalyDetector:
    """Detect anomalies in time series data"""
    def __init__(self, sequence_length=100):
        self.sequence_length = sequence_length
        self.model = Sequential([
            LSTM(64, input_shape=(sequence_length, 1)),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
    
    def prepare_data(self, data):
        """Prepare sequences"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i+self.sequence_length])
            y.append(data[i+self.sequence_length])
        return np.array(X), np.array(y)
    
    def fit(self, data, epochs=50):
        """Train model"""
        X, y = self.prepare_data(data)
        self.model.fit(X, y, epochs=epochs, batch_size=32)
        
        # Calculate reconstruction error distribution
        predictions = self.model.predict(X)
        self.threshold = np.mean(
            np.abs(predictions - y)
        ) + 2 * np.std(np.abs(predictions - y))
    
    def detect_anomalies(self, data):
        """Detect anomalies in new data"""
        X, y = self.prepare_data(data)
        predictions = self.model.predict(X)
        errors = np.abs(predictions - y)
        
        return errors > self.threshold
```

## 4. Audio Applications 🎵

### Speech Recognition
```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import librosa

def create_speech_recognizer():
    """Create speech recognition model"""
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )
    
    return model, tokenizer

def transcribe_audio(audio_path, model, tokenizer):
    """Transcribe audio file"""
    # Load audio
    audio, rate = librosa.load(
        audio_path,
        sr=16000
    )
    
    # Tokenize
    inputs = tokenizer(
        audio,
        return_tensors='pt',
        padding=True
    )
    
    # Get logits
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    
    # Decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])
    
    return transcription
```

## Best Practices for Applications 🌟

1. **Data Preparation**
   ```python
   def prepare_data(data):
       """Prepare data for training"""
       # Handle missing values
       data = data.fillna(method='ffill')
       
       # Scale features
       scaler = StandardScaler()
       data_scaled = scaler.fit_transform(data)
       
       # Split data
       train_size = int(len(data_scaled) * 0.8)
       train = data_scaled[:train_size]
       test = data_scaled[train_size:]
       
       return train, test, scaler
   ```

2. **Model Monitoring**
   ```python
   class ModelMonitor:
       """Monitor model performance"""
       def __init__(self, model):
           self.model = model
           self.metrics_history = []
       
       def log_metrics(self, metrics):
           """Log model metrics"""
           self.metrics_history.append({
               'timestamp': pd.Timestamp.now(),
               **metrics
           })
       
       def check_drift(self, new_data, threshold=0.1):
           """Check for data drift"""
           predictions = self.model.predict(new_data)
           current_metrics = calculate_metrics(predictions)
           
           # Compare with historical metrics
           if abs(current_metrics - self.baseline_metrics) > threshold:
               return True
           return False
   ```

3. **Production Deployment**
   ```python
   def create_production_model():
       """Create model for production"""
       model = create_model()
       
       # Add preprocessing layers
       model = tf.keras.Sequential([
           tf.keras.layers.InputLayer(input_shape=(None,)),
           tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32)),
           tf.keras.layers.Normalization(),
           model
       ])
       
       return model
   ```

## Next Steps 🚀

Congratulations! You've completed the Neural Networks section. Continue exploring other advanced algorithms in the course!
