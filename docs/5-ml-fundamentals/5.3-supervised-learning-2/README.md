# Supervised Learning - Part 2

**After this submodule:** you can use the lessons linked below and complete the exercises that match **Supervised Learning - Part 2** in your course schedule.

## Overview

Part 2 moves from single learners to **ensembles** ([random forest](random-forest/1-introduction.md), [gradient boosting](gradient-boosting/1-introduction.md)), **neural networks** ([introduction](neural-networks/1-introduction.md) plus optional [backpropagation](backpropagation/1-introduction.md) deep dive), and **regularization** ([overview](regularization/1-introduction.md)) to control complexity. **Prerequisites:** [5.2 Supervised learning 1](../5.2-supervised-learning-1/README.md); evaluation ideas from [5.5](../5.5-model-eval/README.md) are worth skimming early so you tune with the right metrics.

## Why this matters

Tree ensembles and boosted models still dominate many tabular leaderboards; neural networks power vision and language. Regularization and honest validation are what keep these flexible models from overfitting noise.

Welcome to the second part of supervised learning! This section covers powerful ensemble methods and neural networks that have revolutionized machine learning. These advanced algorithms build upon the fundamentals you learned in Part 1 to solve even more complex problems.

## Helpful video

Crash Course AI: supervised learning framing (~15 min).

<iframe width="560" height="315" src="https://www.youtube.com/embed/4qVRBYAdLAo" title="Supervised Learning: Crash Course AI" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Learning Objectives

By the end of this section, you will be able to:

1. Understand and implement Random Forests for ensemble learning
2. Master Gradient Boosting techniques with XGBoost, LightGBM, and CatBoost
3. Build and train Neural Networks using modern frameworks
4. Apply regularization techniques to prevent overfitting
5. Choose appropriate algorithms for different problems
6. Tune advanced model parameters effectively

## Algorithms Overview

### 1. [Random Forest](random-forest/1-introduction.md)

Ensemble method using multiple decision trees:

- Bootstrap aggregating (bagging)
- Random feature selection
- Parallel training
- Built-in feature importance

Perfect for:

- High-dimensional data
- Complex non-linear relationships
- Feature importance analysis
- When stability is important

### 2. [Gradient Boosting](gradient-boosting/1-introduction.md)

Sequential ensemble method:

- Builds models iteratively
- Each model corrects previous errors
- Strong predictive power
- Multiple implementations (XGBoost, LightGBM, CatBoost)

Ideal for:

- Structured/tabular data
- Competition-winning performance
- When accuracy is important
- Handling imbalanced data

### 3. [Neural Networks](neural-networks/1-introduction.md)

Deep learning foundation:

- Multiple layers of neurons
- Automatic feature learning
- Various architectures (CNN, RNN, Transformers)
- Transfer learning capabilities

Best for:

- Complex pattern recognition
- Image and video processing
- Natural language processing
- When large data is available

### 4. [Regularization](regularization/1-introduction.md)

Techniques to prevent overfitting:

- L1 regularization (Lasso)
- L2 regularization (Ridge)
- Elastic Net
- Dropout
- Early stopping

## Algorithm Selection Guide

Treat the stub below as a **starting point**; always compare against simpler baselines and validate with cross-validation.

### Classification Tasks

```python
def select_classifier(data_characteristics):
    if data_characteristics.image_data:
        return "Neural Networks (CNN)"
    elif data_characteristics.text_data:
        return "Neural Networks (Transformer)"
    elif data_characteristics.need_interpretability:
        return "Random Forest"
    elif data_characteristics.need_best_accuracy:
        return "Gradient Boosting"
    else:
        return "Try multiple and compare"
```

### Performance Comparison

| Algorithm | Training Speed | Prediction Speed | Interpretability | Memory Usage |
|-----------|---------------|------------------|------------------|--------------|
| Random Forest | Fast | Very Fast | High | Medium |
| Gradient Boosting | Slow | Fast | Medium | Low |
| Neural Networks | Slow | Very Fast | Low | High |

## Prerequisites

Before diving in, ensure you're comfortable with:

1. Python programming
2. Basic machine learning concepts
3. Decision trees (from Part 1)
4. Model evaluation techniques
5. Basic calculus and linear algebra (for neural networks)

## Tools and Libraries

```python
# Essential libraries
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import torch
import xgboost
import lightgbm
```

## Best Practices

1. **Data Preparation**
   - Handle missing values
   - Scale features appropriately
   - Split data properly
   - Create validation set

2. **Model Selection**
   - Start simple, increase complexity
   - Use cross-validation
   - Consider computational resources
   - Think about interpretability needs

3. **Training Process**
   - Monitor training metrics
   - Use early stopping
   - Apply appropriate regularization
   - Save model checkpoints

4. **Evaluation**
   - Use multiple metrics
   - Check for overfitting
   - Analyze feature importance
   - Validate on holdout set

## Common Pitfalls

1. **Random Forest**
   - Too many trees (diminishing returns)
   - Correlated features
   - Class imbalance
   - Memory constraints

2. **Gradient Boosting**
   - Overfitting
   - Too high learning rate
   - Too deep trees
   - Training time

3. **Neural Networks**
   - Vanishing/exploding gradients
   - Overfitting
   - Architecture complexity
   - Hardware requirements

## Ready to Begin?

Start with [Random Forest](random-forest/1-introduction.md) for bagging and feature-importance intuition, then follow the numbered lessons in each subfolder (introduction → math → implementation → advanced → applications). Add the [backpropagation](backpropagation/1-introduction.md) sequence when you want the training-time story spelled out step by step. Each algorithm builds upon previous concepts, so following the suggested order will maximize your learning experience.

The best way to learn this is by doing. Each section includes hands-on examples and exercises to reinforce your understanding.
