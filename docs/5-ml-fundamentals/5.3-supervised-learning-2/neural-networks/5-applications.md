---
reading_minutes: 25
objectives:
  - "Apply neural networks to computer-vision tasks (image classification, object detection) using CNN-style models."
  - "Apply neural networks to NLP tasks (text classification, sequence tagging) using embeddings plus recurrent or transformer encoders."
  - "Apply neural networks to time-series forecasting with sliding windows and recurrent or attention-based models."
---

# Real-World Applications of Neural Networks

**After this lesson:** you can explain the core ideas in “Real-World Applications of Neural Networks” and reproduce the examples here in your own notebook or environment.

## Overview

Where fully connected nets still appear and when to reach for specialized architectures instead.


## Welcome to Neural Network Applications

Neural networks are transforming industries and solving real-world problems every day. In this guide, we'll explore practical applications that you can implement and understand. Think of it like learning to cook different types of cuisine - each application has its own unique flavor and techniques!

## Why Applications Matter

Understanding real-world applications helps you:

- See the practical value of neural networks
- Learn how to adapt techniques to different problems
- Build a portfolio of projects
- Prepare for industry challenges

## 1. Computer Vision: Teaching Computers to See

Computer vision is like giving computers the ability to understand and interpret visual information, just like humans do.

### Image Classification: Identifying Objects

Image classification is like teaching a computer to recognize different types of objects in photos. For example, identifying whether an image contains a cat or a dog.

> **Illustrative reference — not runnable as-is.** Loads ResNet50 ImageNet weights (auto-downloaded) and expects a `train_dir/` of labelled images. Study the transfer-learning structure rather than executing it; for a network you can run end-to-end, see the [MNIST demo](1-introduction.md).

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_image_classifier(num_classes):
    """Create a model to classify images into different categories"""
    # Load pre-trained ResNet50 model
    base_model = ResNet50(
        weights='imagenet',  # Use weights trained on ImageNet
        include_top=False,   # Don't include the final classification layer
        input_shape=(224, 224, 3)  # Standard image size
    )

    # Freeze the pre-trained layers
    base_model.trainable = False

    # Add our custom classification layers
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),  # Convert features to vector
        tf.keras.layers.Dense(256, activation='relu'),  # Add a dense layer
        tf.keras.layers.Dropout(0.5),  # Prevent overfitting
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer
    ])

    return model

# Prepare data with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=20,  # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift width
    height_shift_range=0.2,  # Randomly shift height
    horizontal_flip=True,  # Randomly flip horizontally
    fill_mode='nearest'  # How to fill empty spaces
)

# Create data generator
train_generator = train_datagen.flow_from_directory(
    'train_dir',  # Directory containing training images
    target_size=(224, 224),  # Resize images
    batch_size=32,  # Number of images per batch
    class_mode='categorical'  # Type of classification
)

# Create and train model
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
        tf.keras.callbacks.EarlyStopping(patience=3)  # Stop if no improvement
    ]
)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-3" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports</span>
    </div>
    <div class="code-callout__body">
      <p>TensorFlow, the pre-trained ResNet50 backbone, and <code>ImageDataGenerator</code> for on-the-fly augmentation are imported to set up transfer learning.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="5-26" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Transfer Learning Model</span>
    </div>
    <div class="code-callout__body">
      <p>ResNet50 is loaded with frozen ImageNet weights; a custom head (GlobalAveragePooling → Dense 256 → Dropout → softmax) is appended for the target number of classes.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="28-45" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Data Augmentation</span>
    </div>
    <div class="code-callout__body">
      <p><code>ImageDataGenerator</code> normalizes pixels and applies random rotations, shifts, and flips on the fly; <code>flow_from_directory</code> streams batches directly from a folder of labeled subdirectories.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="47-61" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Compile and Train</span>
    </div>
    <div class="code-callout__body">
      <p>Adam with categorical cross-entropy loss is used; early stopping with <code>patience=3</code> halts training automatically when validation accuracy stops improving.</p>
    </div>
  </div>
</aside>
</div>

### Object Detection: Finding and Locating Objects

Object detection is like teaching a computer to not only recognize objects but also find where they are in an image. This is useful for applications like self-driving cars or security systems.

> **Illustrative reference — not runnable as-is.** Needs the `ultralytics` package, a `data.yaml` dataset config, and a YOLO-weights download on first run.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from ultralytics import YOLO

def create_object_detector():
    """Create a YOLO object detector"""
    # Load pre-trained YOLO model
    model = YOLO('yolov8n.pt')  # Small version for faster training

    # Train the model
    model.train(
        data='data.yaml',  # Configuration file
        epochs=100,        # Number of training cycles
        imgsz=640,         # Image size
        batch=16,          # Batch size
        save=True          # Save the model
    )

    return model

def detect_objects(model, image):
    """Detect objects in an image"""
    # Run detection
    results = model(image)

    # Process results
    boxes = results[0].boxes
    for box in boxes:
        # Get coordinates of detected object
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        class_id = box.cls[0]

        print(f"Found: {class_id} with {confidence:.2f} confidence")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-17" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Train YOLO</span>
    </div>
    <div class="code-callout__body">
      <p><code>YOLO('yolov8n.pt')</code> loads the nano variant's pretrained weights; <code>model.train</code> fine-tunes on a custom dataset described in <code>data.yaml</code> for 100 epochs at 640-px resolution.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="19-32" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Detect Objects</span>
    </div>
    <div class="code-callout__body">
      <p><code>detect_objects</code> runs inference on a single image and iterates over detected bounding boxes, printing the class ID and confidence score for each object found.</p>
    </div>
  </div>
</aside>
</div>

## 2. Natural Language Processing: Understanding Text

NLP is like teaching computers to understand and work with human language, from simple tasks like sentiment analysis to complex ones like translation.

### Text Classification: Understanding Sentiment

Text classification helps computers understand the meaning or sentiment of text, like determining if a product review is positive or negative.

> **Illustrative reference — not runnable as-is.** Needs the `transformers` package and downloads `bert-base-uncased` (~400 MB).

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

def create_text_classifier(num_labels):
    """Create a BERT model for text classification"""
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels
    )

    return model, tokenizer

def prepare_text_data(texts, tokenizer, max_length=128):
    """Prepare text for BERT model"""
    return tokenizer(
        texts,
        padding=True,      # Add padding to make all sequences same length
        truncation=True,   # Cut off text if too long
        max_length=max_length,
        return_tensors='tf'  # Return TensorFlow tensors
    )

# Example: Sentiment Analysis
texts = [
    "This product is amazing! I love it!",
    "Terrible customer service, would not recommend.",
    "It's okay, nothing special."
]
labels = [1, 0, 2]  # Positive, Negative, Neutral

# Create and train model
model, tokenizer = create_text_classifier(num_labels=3)
inputs = prepare_text_data(texts, tokenizer)

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
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-13" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Load BERT</span>
    </div>
    <div class="code-callout__body">
      <p><code>create_text_classifier</code> loads the <code>bert-base-uncased</code> tokenizer and its TensorFlow sequence-classification head with <code>num_labels</code> output units.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="15-23" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Tokenize Text</span>
    </div>
    <div class="code-callout__body">
      <p><code>prepare_text_data</code> pads, truncates, and encodes a list of strings into input-ID and attention-mask tensors that BERT expects.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="25-49" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Compile and Fine-tune</span>
    </div>
    <div class="code-callout__body">
      <p>A tiny three-sentence sentiment dataset is tokenized, then the model is compiled with a low learning rate (2e-5 is standard for BERT fine-tuning) and sparse categorical cross-entropy, then trained for 3 epochs.</p>
    </div>
  </div>
</aside>
</div>

**Representative output** (illustrative — these snippets are not executed during the site build). After fine-tuning, classifying the three example reviews would look like:

```text
"This product is amazing! I love it!"              -> Positive (0.97)
"Terrible customer service, would not recommend."  -> Negative (0.95)
"It's okay, nothing special."                      -> Neutral  (0.86)
```

### Machine Translation: Breaking Language Barriers

Machine translation helps computers translate text from one language to another, like having a digital translator in your pocket.

> **Illustrative reference — not runnable as-is.** Needs the `transformers` package and downloads the Helsinki-NLP MarianMT weights.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from transformers import MarianMTModel, MarianTokenizer

def create_translator(src_lang='en', tgt_lang='fr'):
    """Create a machine translation model"""
    # Load pre-trained translation model
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    return model, tokenizer

def translate_text(text, model, tokenizer):
    """Translate text from source to target language"""
    # Prepare text for model
    inputs = tokenizer(text, return_tensors='pt', padding=True)

    # Generate translation
    outputs = model.generate(**inputs)

    # Convert back to text
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translation

# Example usage
model, tokenizer = create_translator('en', 'fr')
text = "Hello, how are you today?"
translation = translate_text(text, model, tokenizer)
print(f"English: {text}")
print(f"French: {translation}")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-11" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Load Translator</span>
    </div>
    <div class="code-callout__body">
      <p><code>create_translator</code> builds the Helsinki-NLP model name from the language pair and loads the matching Marian tokenizer and model; swapping language codes selects a different translation direction.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-23" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Translate Text</span>
    </div>
    <div class="code-callout__body">
      <p>Input text is tokenized to PyTorch tensors, <code>model.generate</code> runs beam search to produce the most likely translation, and <code>tokenizer.decode</code> converts token IDs back to a readable string.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="25-31" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Usage Demo</span>
    </div>
    <div class="code-callout__body">
      <p>An English greeting is translated to French; the original and translation are printed side by side to verify the output.</p>
    </div>
  </div>
</aside>
</div>

**Representative output** (illustrative — not executed during the site build):

```text
English: Hello, how are you today?
French: Bonjour, comment allez-vous aujourd'hui ?
```

## 3. Time Series Analysis: Predicting the Future

Time series analysis helps computers understand and predict patterns in data that changes over time, like stock prices or weather patterns.

### Stock Price Prediction

Predicting stock prices is like trying to forecast the weather - we use past patterns to predict future movements.

> **Illustrative reference — not runnable as-is.** Needs the `yfinance` package and live internet access to fetch AAPL prices.

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import yfinance as yf

def create_stock_predictor(sequence_length):
    """Create an LSTM model for stock prediction"""
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
        Dense(1)  # Predict next day's price
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_sequences(data, sequence_length):
    """Prepare time series data for LSTM"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

# Get stock data
stock = yf.Ticker('AAPL')
data = stock.history(period='1y')['Close'].values

# Normalize data
data = (data - data.mean()) / data.std()

# Prepare sequences
sequence_length = 10  # Use 10 days to predict next day
X, y = prepare_sequences(data, sequence_length)

# Create and train model
model = create_stock_predictor(sequence_length)
model.fit(X, y, epochs=50, batch_size=32)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-4" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Imports</span>
    </div>
    <div class="code-callout__body">
      <p>NumPy, Keras LSTM/Dense layers, and <code>yfinance</code> for fetching real stock prices are imported to build a minimal time-series predictor.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="6-14" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">LSTM Model</span>
    </div>
    <div class="code-callout__body">
      <p>A single LSTM layer with 50 units processes a window of past prices; a Dense(1) output head predicts the next day's value, compiled with MSE loss for regression.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="16-22" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Sequence Prep</span>
    </div>
    <div class="code-callout__body">
      <p><code>prepare_sequences</code> slides a window of <code>sequence_length</code> over the normalized price series, creating input-output pairs where each target is the immediately following price.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="24-36" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Fetch and Train</span>
    </div>
    <div class="code-callout__body">
      <p>One year of AAPL closing prices is downloaded, z-score normalized, windowed into sequences, and used to train the LSTM for 50 epochs.</p>
    </div>
  </div>
</aside>
</div>

## Common Mistakes to Avoid

1. **Using Too Complex Models**
   - Start with simple architectures
   - Only add complexity when needed
   - Monitor performance improvements

2. **Data Preparation Issues**
   - Always preprocess your data
   - Handle missing values
   - Normalize features appropriately

3. **Overfitting**
   - Use validation data
   - Implement early stopping
   - Consider regularization techniques

## Practical Tips

1. **Start Small**
   - Begin with simple problems
   - Use pre-trained models when possible
   - Gradually increase complexity

2. **Data Quality**
   - Clean and preprocess data
   - Use appropriate augmentation
   - Handle class imbalance

3. **Model Evaluation**
   - Use appropriate metrics
   - Compare with baselines
   - Consider business impact

## Next Steps

Ready to build your own applications? Try these projects:

1. Create an image classifier for your own dataset
2. Build a sentiment analyzer for product reviews
3. Develop a stock price predictor
4. Implement a language translator

Remember, the best way to learn is by doing! Start with a simple project and gradually add complexity as you become more comfortable with the concepts.

## Gotchas

- **Applying `preprocess_input` for ResNet50 to data that was already normalized to [0, 1]** — `ResNet50`'s `preprocess_input` expects pixel values in [0, 255] and subtracts ImageNet channel means. If you rescale to [0, 1] first (e.g., `rescale=1./255` in `ImageDataGenerator`) and then also call `preprocess_input`, the resulting values are in the range [-2, -1] rather than the expected [-124, 151], silently degrading accuracy.
- **Fine-tuning BERT with a learning rate above 5e-5 destroys pretrained weights** — The example uses `learning_rate=2e-5`, which is standard for BERT fine-tuning. Using a default Adam rate of 1e-3 or 1e-4 causes catastrophic forgetting: the pretrained representations are overwritten within the first few batches, and the model performs no better than a randomly initialized one.
- **LSTM with `activation='relu'` can produce exploding hidden states** — The stock predictor uses `LSTM(50, activation='relu')`. ReLU is not the standard LSTM activation (tanh is). With ReLU, large inputs can cause the hidden state to grow unboundedly across timesteps since there is no saturation. Use the default `activation='tanh'` unless you have a specific reason and gradient clipping in place.
- **Walk-forward training leaks future data if the target shift is wrong** — `create_stock_features` sets `Target = df['Close'].shift(-1) / df['Close'] - 1`. Using this dataframe directly without careful index alignment can include tomorrow's price in today's feature row (look-ahead bias), producing unrealistically high backtest performance.
- **`flow_from_directory` infers class labels from subdirectory names, not a CSV** — The image classification example assumes images live in subdirectories named by class. If your data is organized differently (e.g., a flat folder with a labels CSV), `flow_from_directory` will silently treat the entire folder as one class. Use `flow_from_dataframe` instead.
- **`YOLO('yolov8n.pt')` downloads weights on first run and requires internet access** — In restricted environments (e.g., corporate networks, cloud VMs without egress), this call will hang or fail with a confusing timeout error. Download the checkpoint manually and pass the local path instead.
