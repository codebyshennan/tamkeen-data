---
reading_minutes: 15
objectives:
  - "Explain why naive Bayes is \"naive\": features are assumed conditionally independent given the class."
  - "Identify the three classic NB applications (spam, news categorisation, medical screening) and what makes them NB-friendly."
  - "Map any NB classifier to two phases: learn class priors and feature likelihoods, then predict by posterior maximisation."
---
# Introduction to Naive Bayes Classification

**After this lesson:** you can explain Introduction to Naive Bayes Classification and try the examples in your own notebook.

## Overview

**Naive Bayes** classifiers use **Bayes' theorem** and assume features are **conditionally independent** given the label, fast, closed-form training and strong baselines for text and categorical data. **Prerequisites:** [5.2 README](../README.md); probability notation from [Module 4](../../../4-stat-analysis/README.md) helps before [2-math-foundation.md](2-math-foundation.md).


## Welcome to Naive Bayes

Before we dive into the technical details, get clear on why Naive Bayes is such an important algorithm in machine learning. Think of it as your first "smart assistant" in the world of classification - it's simple yet powerful, making it perfect for beginners to understand and use.

## What is Naive Bayes?

![Bayes' Theorem visualised as overlapping sets: prior event A, evidence B, and their intersection A∩B](assets/bayes_theorem_venn.png)

Imagine you're trying to sort your emails into "spam" and "not spam" folders. You don't need to read every word carefully - you just look for certain clues like "free", "money", or "win". Naive Bayes works similarly - it's a smart way to make decisions based on probabilities and patterns.

### Breaking Down the Name

Understand what "Naive Bayes" means:

1. **Bayes**: Named after Reverend Thomas Bayes, who created a way to update probabilities when we get new information. Think of it like updating your opinion when you learn new facts.

2. **Naive**: This might sound negative, but it's actually a clever simplification. The algorithm assumes that different features (like words in an email) don't affect each other. It's like saying "the presence of the word 'free' doesn't tell us anything about whether the word 'money' will appear."

### Why This Matters

Naive Bayes is special because:

- It's fast and efficient
- Works well with small datasets
- Easy to understand and implement
- Great for text classification
- Perfect for real-time predictions

## Real-World Applications

look at some everyday examples where Naive Bayes is used:

### 1. Email Spam Detection

- How it works: Analyzes words in emails to decide if they're spam
- Why it's good: Fast and accurate, even with simple features
- Real impact: Saves you from unwanted emails every day

### 2. Medical Diagnosis

- How it works: Uses symptoms and test results to predict diseases
- Why it's good: Can handle multiple symptoms at once
   - Teaching use: a toy screening pattern for discussing probabilities, not a deployable clinical system

### 3. News Article Categorization

- How it works: Classifies articles into topics like sports, politics, or technology
- Why it's good: Works well with text data
- Real impact: Helps organize and recommend relevant news

### How Naive Bayes Classifies: The Two Phases

{% include mermaid-diagram.html src="5-ml-fundamentals/5.2-supervised-learning-1/naive-bayes/diagrams/1-introduction-1.mmd" %}

*The "naive" assumption: multiplying individual word probabilities together as if they are independent. This is rarely true in practice, but it works surprisingly well for text classification.*

## The Learning Journey Ahead

In this course, we'll explore Naive Bayes step by step:

1. **Mathematical Foundation** (Next Section)
   - Learn the basic probability concepts
   - Understand how Bayes' Theorem works
   - See how probabilities help make decisions

2. **Types of Naive Bayes**
   - Discover different versions for different data types
   - Learn when to use each type
   - See real examples of each type

3. **Implementation**
   - Write your first Naive Bayes code
   - Work with real datasets
   - Build your own classifiers

4. **Advanced Topics**
   - Learn professional tips and tricks
   - Handle real-world challenges
   - Optimize your models

## Common Questions from Beginners

### "Is Naive Bayes really naive?"

Yes, but in a good way! The "naive" assumption makes the algorithm simpler and faster, while still being surprisingly effective.

### "Do I need to be good at math?"

Basic probability knowledge helps, but we'll explain everything step by step. The focus is on understanding the concepts, not complex calculations.

### "What can I build with Naive Bayes?"

You can create:

- Spam filters
- Sentiment analyzers
- Document classifiers
- Medical diagnosis tools
- And much more!

## Gotchas

- **Thinking "naive" means the model is inaccurate**: The independence assumption is almost always violated in practice (words in text are highly correlated, symptoms co-occur), yet Naive Bayes often rivals much more complex models on text classification. The model is naive about dependencies but not about overall performance.
- **Applying Naive Bayes to regression problems**: Naive Bayes is a probabilistic classifier, not a regressor. Using it to predict continuous outputs (like house prices) is not directly possible with `GaussianNB`, `MultinomialNB`, or `BernoulliNB` in scikit-learn. For regression, use Gaussian Process regression or other methods.
- **Confusing the prior with class frequency in a balanced training set**: Even if you construct a balanced training dataset (equal class sizes), the real-world prior may be very different (e.g., spam is 1% of email). Forcing a 50/50 prior by balancing your training data bakes incorrect priors into the model; pass `class_prior` explicitly if the true class distribution differs from your training distribution.
- **Expecting Naive Bayes to work out-of-the-box on mixed feature types**: scikit-learn's NB variants each require a specific input type: `GaussianNB` for continuous, `MultinomialNB` for non-negative counts, `BernoulliNB` for binary. Passing the wrong feature type (e.g., negative values to `MultinomialNB`) raises errors or produces silently wrong probability estimates.
- **Forgetting that Naive Bayes cannot learn feature interactions**: Because of the independence assumption, Naive Bayes will miss patterns that only emerge when two features occur together (e.g., "not good" having opposite sentiment to "good"). When such interactions are important, logistic regression or tree-based models will outperform it.

## Next Steps

Ready to dive deeper? Start with the [Mathematical Foundation](2-math-foundation.md) to understand how Naive Bayes makes its predictions.

Remember: Every expert was once a beginner. Take your time, ask questions, and enjoy the learning journey!
