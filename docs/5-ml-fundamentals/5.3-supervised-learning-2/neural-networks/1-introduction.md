---
reading_minutes: 20
objectives:
  - "Define a neural network as stacked differentiable layers and connect units → layers → loss → optimizer into one mental model."
  - "Pick a network family for a problem: MLP for tabular, CNN for images, RNN/Transformer for sequences."
  - "Recognise where neural networks beat classical ML (large datasets, raw inputs, complex feature interactions) and where they don't (small tabular, hard interpretability, tight latency)."
---

# Introduction to Neural Networks

**After this lesson:** you can explain Introduction to Neural Networks and try the examples in your own notebook.

## Overview

**Neural networks** stack differentiable layers of units; training adjusts **weights** to reduce a **loss** (usually via gradient-based optimization, see [backpropagation](../backpropagation/1-introduction.md)). **Prerequisites:** vectors and matrices from Module 1; [5.3 README](../README.md).


## Welcome to Neural Networks

Imagine you're teaching a child to recognize different types of fruits. At first, they might make mistakes, but with practice and feedback, they get better. Neural networks learn in a similar way! They're computer systems inspired by how our brains work, designed to learn from examples and improve over time.

## What are Neural Networks?

Think of a neural network like a team of experts working together to solve a puzzle. Each expert (neuron) specializes in recognizing different patterns, and they communicate with each other to reach a final decision.

### Why This Matters

Neural networks power many of the technologies we use daily:

- Your phone's face recognition
- Smart assistants like Siri or Alexa
- Email spam filters
- Medical diagnosis systems
- Self-driving cars

### Key Concepts Explained Simply

1. **Neurons (Nodes)**
   - Like tiny decision-makers in your brain
   - Each neuron looks at information and decides whether to "fire" or not
   - Example: A neuron might help decide if an image contains a cat

2. **Layers**
   - Think of layers like a factory assembly line
   - Input Layer: Receives raw data (like a photo)
   - Hidden Layers: Process and transform the data
   - Output Layer: Gives the final answer

3. **Connections**
   - Like roads between cities
   - Weights: How important each connection is
   - Biases: Like adjusting the difficulty level

{% include mermaid-diagram.html src="5-ml-fundamentals/5.3-supervised-learning-2/neural-networks/diagrams/1-introduction-1.mmd" %}

*Each arrow carries a **weight**. During training, weights are adjusted so the output ŷ gets closer to the true label.*

## When to Use Neural Networks?

### Perfect For

- Complex pattern recognition (like identifying objects in photos)
- Understanding human language (chatbots, translation)
- Predicting future trends (stock prices, weather)
- Creating art and music
- Playing games (like chess or Go)

### Less Suitable For

- Small datasets (like less than 100 examples)
- When you need to explain exactly how a decision was made
- If you're working with a slow computer
- Simple problems that can be solved with basic math
- When you need instant results

## Types of Neural Networks

### 1. Feedforward Neural Networks

- Like a one-way street for information
- Great for: Predicting house prices, customer preferences
- Example: Netflix recommending movies you might like

### 2. Convolutional Neural Networks (CNN)

- Specialized for images and videos
- Like having a magnifying glass that looks for patterns
- Used in: Face recognition, medical imaging, self-driving cars

### 3. Recurrent Neural Networks (RNN)

- Good at understanding sequences
- Like reading a book and remembering the story
- Used in: Speech recognition, predicting text, music generation

### 4. Long Short-Term Memory (LSTM)

- Advanced version of RNN
- Better at remembering important information
- Used in: Language translation, weather forecasting

{% include mermaid-diagram.html src="5-ml-fundamentals/5.3-supervised-learning-2/neural-networks/diagrams/1-introduction-2.mmd" %}

*Use this as a starting heuristic, real projects often combine types (e.g. CNN + LSTM for video captioning).*

## Common Mistakes to Avoid

1. **Using too complex models for simple problems**
   - Start simple and only add complexity when needed
   - Example: Don't use a deep network to predict if a number is even or odd

2. **Not enough data**
   - Neural networks need lots of examples to learn
   - Rule of thumb: At least 1000 examples per class

3. **Forgetting to normalize data**
   - Like comparing apples and oranges
   - Always scale your data to similar ranges

4. **Training for too long**
   - Can lead to memorizing instead of learning
   - Use validation data to check when to stop

## Getting Started with Code

Build a simple neural network to recognize handwritten digits. This is like teaching a computer to read numbers!

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import tensorflow as tf
from tensorflow import keras

# Load the famous MNIST dataset of handwritten digits
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images (scale pixel values to 0-1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Create a simple neural network
model = keras.Sequential([
    # Flatten the 28x28 images into a single row
    keras.layers.Flatten(input_shape=(28, 28)),
    # First hidden layer with 128 neurons
    keras.layers.Dense(128, activation='relu'),
    # Output layer with 10 neurons (one for each digit 0-9)
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.3f}")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-10" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Load and Normalize MNIST</span>
    </div>
    <div class="code-callout__body">
      <p>Load the 70,000 handwritten digit images from Keras; divide by 255 to scale pixel values from [0, 255] to [0, 1], neural networks train faster and more stably on normalized inputs.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="12-26" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Build and Compile</span>
    </div>
    <div class="code-callout__body">
      <p>The <code>Sequential</code> model flattens each 28×28 image to 784 inputs, passes through a 128-unit ReLU hidden layer, then outputs 10 softmax probabilities (one per digit); Adam optimizer minimizes cross-entropy.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="28-35" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Train and Evaluate</span>
    </div>
    <div class="code-callout__body">
      <p>Train for 5 epochs then evaluate on the held-out test set; <code>test_acc</code> reflects how well the network generalizes to unseen digit images.</p>
    </div>
  </div>
</aside>
</div>

## Gotchas

- **Using `sparse_categorical_crossentropy` when labels are already one-hot encoded**: The MNIST example uses integer labels (0-9) with `sparse_categorical_crossentropy`, which is correct. If you one-hot encode the labels first and then use this loss, Keras interprets each label as a class index into a 10-class one-hot vector, producing wrong shapes or silent misclassification.
- **Normalizing test data with statistics from the test set**: The example divides by 255 (a known constant), which is safe. For other datasets, always compute mean and std from the training split only and apply those same values to test data. Fitting normalization on test data leaks information and inflates reported accuracy.
- **Choosing network type based on familiarity, not problem type**: The type selector diagram is a starting heuristic. A common mistake is defaulting to a feedforward network for all tasks: CNNs are necessary for spatial data (images), and LSTMs/Transformers are necessary for sequences. Using a dense-only network on images treats each pixel as independent and loses spatial structure.
- **Epoch count of 5 is rarely enough for production**: The introductory MNIST example trains for 5 epochs and achieves decent accuracy on a well-conditioned dataset. For harder datasets or transfer-learning fine-tuning, 5 epochs will underfit. Always monitor validation loss and let early stopping determine the right epoch count.
- **Assuming more hidden neurons always helps**: Adding neurons increases capacity but also training time, memory, and overfitting risk. Neural networks for small datasets (< 1000 examples per class) need regularization (dropout, L2) far more than they need wider layers.

## Additional Resources

### For Beginners

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - Free online book
- [3Blue1Brown's Neural Networks Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - Great visual explanations
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials) - Official beginner guides

### For Practice

- [Kaggle](https://www.kaggle.com/learn/intro-to-deep-learning) - Hands-on exercises
- [Google Colab](https://colab.research.google.com/) - Free cloud notebooks to try code

## Next Steps

Ready to understand the math behind neural networks? Continue to [Mathematical Foundation](2-math-foundation.md) to learn how these amazing systems actually work!
