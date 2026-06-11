---
reading_minutes: 6
objectives:
  - "Distinguish **bias** (error from over-simplifying assumptions) from **variance** (error from sensitivity to training data) and place a model on the underfit–overfit axis."
  - "Recognise the tradeoff: low-bias models tend to be high-variance and vice versa, so total error has a U-shape in model complexity."
  - "Diagnose which side of the tradeoff a model is on by comparing **train vs validation error** — close and high = bias; large gap = variance."
  - "Counter high bias with more capacity / better features; counter high variance with regularisation, more data, or simpler models."
---

# Bias-Variance Tradeoff

**After this lesson:** you can explain the core ideas in “Bias-Variance Tradeoff” and reproduce the examples here in your own notebook or environment.

## Overview

Evaluation-focused view of **bias–variance** and what different validation designs reveal.

Connects to [5.1 bias–variance](../5.1-intro-to-ml/bias-variance.md) intuition.


## Introduction

Understanding the bias-variance tradeoff is crucial in machine learning. It helps us diagnose model performance and make better decisions about model complexity.

## What is Bias?

Bias is the error introduced by approximating a real-world problem with a simplified model. It represents how far off our model's predictions are from the true values, on average.

## What is Variance?

Variance refers to the model's sensitivity to fluctuations in the training data. It measures how much our model's predictions vary when trained on different datasets.

## The Tradeoff

Finding the right balance between bias and variance is key to creating effective models. Too much bias leads to underfitting, while too much variance leads to overfitting.

{% include mermaid-diagram.html src="5-ml-fundamentals/5.5-model-eval/diagrams/bias-variance-1.mmd" %}

## Practical Examples

Let's look at some real-world examples:

1. Linear Regression (High Bias, Low Variance)
   - Simple model
   - Consistent predictions
   - May miss complex patterns

2. Deep Neural Networks (Low Bias, High Variance)
   - Complex model
   - Flexible predictions
   - May overfit to noise

3. Random Forests (Balanced)
   - Ensemble method
   - Combines multiple trees
   - Often achieves good balance

## Best Practices

1. Use cross-validation to assess model performance
2. Monitor learning curves to detect bias/variance issues
3. Try different model complexities
4. Use regularization techniques
5. Consider ensemble methods

## Gotchas

- **Conflating training error with bias** — A model can have near-zero training error and still have high bias if the training set is tiny; always check validation error, not just training error, to diagnose bias.
- **Assuming more data always fixes variance** — Adding data reduces variance only when the model is actually overfitting; if both train and validation error are high (high bias), more data will not help and you need a more complex model.
- **Treating regularization and bias as independent** — Adding L1/L2 regularization to reduce variance simultaneously increases bias; the tradeoff is intentional, but learners often add regularization expecting only gains without realising accuracy on the training set will drop.
- **Ignoring variance in the error estimate itself** — A single train/test split gives one variance sample; use cross-validation to see whether the gap between train and validation scores is stable or fluctuates, which reveals variance in the *evaluation*, not just the model.
- **Confusing irreducible error with high bias** — Even a perfect model cannot beat noise inherent in the data; diagnosing "high bias" incorrectly as irreducible error (or vice versa) leads to wasted complexity tuning.
- **Using only model complexity as the bias-variance dial** — Dataset size, feature engineering quality, and regularization strength all shift the tradeoff; fixing overfitting by switching to a simpler model family often discards useful capacity that regularization could preserve.

## Additional Resources

1. Scikit-learn documentation
2. Research papers on model complexity
3. Online tutorials on bias-variance tradeoff
