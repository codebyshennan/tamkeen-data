---
reading_minutes: 20
objectives:
  - >-
    State and apply Bayes' theorem to compute a posterior class probability from
    priors and likelihoods.
  - >-
    Show why the independence assumption lets us multiply per-feature
    likelihoods (sum log-likelihoods in practice) for closed-form scoring.
  - >-
    Score a document or feature row against multiple classes and pick `argmax`
    as the prediction.
  - >-
    Estimate Gaussian likelihoods for continuous features and recognise when
    this assumption breaks.
---

# Mathematical Foundation of Naive Bayes

**After this lesson:** you can explain Mathematical Foundation of Naive Bayes and try the examples in your own notebook.

## Overview

Walks through **Bayes' theorem**, likelihoods under the independence assumption, and log-prob scoring for classification.

[Introduction](1-introduction.md); probability from Module 4 smooths the notation.

## Welcome to the Math Behind Naive Bayes

Don't worry if math isn't your strongest suit! We'll break down the concepts into simple, understandable pieces. Think of this as learning a new language - we'll start with the basics and build up gradually.

## Understanding Probability: The Language of Naive Bayes

### What is Probability?

Probability is just a fancy way of saying "how likely something is to happen." For example:

* The probability of flipping a coin and getting heads is 50%
* The probability of rolling a 6 on a die is about 16.7%

In Naive Bayes, we use probability to make predictions. It's like being a weather forecaster who says, "There's a 70% chance of rain tomorrow."

### Bayes' Theorem: The Heart of Naive Bayes

Imagine you're a detective trying to solve a case. You have some initial hunches (prior knowledge), and as you gather new evidence, you update your beliefs. That's exactly what Bayes' Theorem does!

#### The Basic Formula

Break down the formula step by step:

\\\[P(y|X) = \frac{P(X|y)P(y)}{P(X)}\\]

Think of it like this:

* \\(P(y|X)\\): "What's the probability of y given X?" (Your updated belief)
* \\(P(X|y)\\): "How likely is X if y is true?" (The evidence)
* \\(P(y)\\): "What was your initial belief about y?" (Your prior knowledge)
* \\(P(X)\\): "How likely is X in general?" (The overall evidence)

![Conditional probability of 'FREE' appearing in spam vs ham emails](../../../../.gitbook/assets/conditional_probability.png)

### Real-World Example: Email Spam Detection

Make this concrete with an email example:

#### Spam posterior from counts

```python
# Suppose we have 1000 emails in our training data
total_emails = 1000
spam_emails = 300        # 300 are spam
emails_with_word_free = 400  # 400 contain "free"
spam_with_word_free = 240    # 240 spam emails contain "free"

# Calculate probabilities
prior = spam_emails / total_emails  # 30% of emails are spam
likelihood = spam_with_word_free / spam_emails  # 80% of spam has "free"
evidence = emails_with_word_free / total_emails  # 40% of all emails have "free"

# Calculate the probability that an email is spam if it contains "free"
posterior = (likelihood * prior) / evidence  # 60% chance it's spam
```

This means:

* If you see an email with the word "free", there's a 60% chance it's spam
* The algorithm learned this from looking at past emails
* It updates its belief based on what it sees

## The "Naive" Assumption: Why It Works

### Understanding Feature Independence

The "naive" part comes from assuming that features do not affect each other. Use a cooking analogy:

Imagine you're making a cake. The recipe says you need:

* Flour
* Sugar
* Eggs
* Butter

The naive assumption is like saying:

* Adding more flour doesn't change how much sugar you need
* Adding eggs doesn't affect how much butter you need

In reality, these ingredients do interact, but assuming they don't makes the math much simpler!

### Why This Simplification Works

Even though features often do affect each other:

1. The simplification makes calculations much faster
2. It often works surprisingly well in practice
3. We care more about getting the right answer than having perfect probabilities

## Making Predictions: The Classification Rule

### How Naive Bayes Makes Decisions

To classify something (like an email as spam or not spam):

1. Calculate the probability for each possible class
2. Choose the class with the highest probability

It's like a voting system where each feature gets a say, and the class with the most votes wins!

### Example: Document Classification

Classify a document as either tech or sports:

#### Tech vs sports score (naive product)

Priors

Equal class priors (50/50) represent a balanced corpus; in a real classifier these would be estimated from training label frequencies.

Likelihoods

Per-word conditional probabilities are set manually to illustrate that tech-domain words appear far less often in sports documents, the core assumption the Naive Bayes classifier exploits.

Score Comparison

Each score multiplies the prior by the three word likelihoods (naive independence assumption); the much higher `tech_score` wins the argmax and the document is labelled "tech".

## Handling Different Types of Data

### Numerical Features: The Gaussian Approach

When dealing with numbers (like height or temperature), we use the Gaussian (Normal) distribution. Think of it as a bell curve that shows how likely different values are.

For example, if we're predicting gender based on height:

* Most men are around 175cm
* Most women are around 162cm
* The curve shows how likely other heights are

#### Gaussian likelihood for height

Distribution Parameters

Mean and standard deviation are specified for each class; these would normally be estimated from training data via maximum likelihood (sample mean and std).

Gaussian PDF and Classify

The function implements the Gaussian PDF formula; calling it at height=168 for both classes gives two density values, the class with the higher value is predicted, demonstrating the Gaussian Naive Bayes decision rule.

## Common Mistakes to Avoid

1. **Forgetting to Scale Numerical Features**
   * Always scale your numbers (like height, weight) before using Gaussian Naive Bayes
   * Use tools like StandardScaler from scikit-learn
2. **Ignoring the Prior Probabilities**
   * If your classes are imbalanced (e.g., 90% not spam, 10% spam), account for this
   * Use class\_prior parameter in scikit-learn
3. **Using the Wrong Type of Naive Bayes**
   * Use Gaussian for numbers
   * Use Multinomial for counts (like word frequencies)
   * Use Bernoulli for yes/no features

## Practice Makes Perfect

The best way to understand these concepts is to practice:

1. Try implementing a simple spam detector
2. Experiment with different types of features
3. Compare the results with and without scaling
4. See how the algorithm behaves with different datasets

## Gotchas

* **Numeric underflow from multiplying raw probabilities**: The `tech_score` and `sports_score` example multiplies three small probabilities together. With real documents containing dozens of words, the product quickly underflows to `0.0` in floating-point, making it impossible to compare classes. Always work in log-space (`log(P(y)) + Σ log(P(xi|y))`) in production code; scikit-learn does this internally via `predict_log_proba`.
* **Ignoring the zero-probability problem before seeing smoothing**: If a word in the test document never appeared in training data for a class, the entire product for that class becomes zero, regardless of all other evidence. The `alpha=1.0` (Laplace smoothing) parameter in scikit-learn handles this, but the manual calculation in the lesson does not, so learners reproducing it on real data will get zero scores for unseen vocabulary.
* **Confusing likelihood and posterior**: `P(X|y)` (likelihood) and `P(y|X)` (posterior) are different quantities. A common mistake is to compare raw likelihoods across classes without multiplying by the prior, which gives wrong answers whenever class frequencies are not equal. The Bayes formula is required to get the correct posterior.
* **Misreading the `evidence` term as needing per-class computation**: `P(X)` (the evidence denominator) is the same for all classes, so for classification purposes it is a constant that cancels out when comparing `P(y|X)` across classes. Students sometimes compute a different evidence for each class, leading to probabilities that don't sum to 1.
* **Treating the Gaussian PDF value as a probability**: The `gaussian_probability` function returns a probability density, not a probability. Densities can exceed 1 for narrow distributions. The decision rule (comparing densities) is correct, but quoting the raw density value as "a 0.08 probability" in an explanation is technically wrong.

## Next Steps

Ready to see these concepts in action? Move on to [Types of Naive Bayes](3-types.md) to learn about the different versions of the algorithm and when to use each one.

Remember: Math is just a tool to help us make better predictions. Focus on understanding the concepts, and the formulas will make more sense!
