---
reading_minutes: 20
objectives:
  - "Pick `GaussianNB`, `MultinomialNB`, or `BernoulliNB` from the shape of the input features (continuous, counts, or binary)."
  - "Match each variant to a canonical use case: medical readings, news classification, or yes/no spam features."
  - "Use the lesson's decision table to make the right pick on unfamiliar data."
---
# Types of Naive Bayes Classifiers

**After this lesson:** you can explain Types of Naive Bayes Classifiers and try the examples in your own notebook.


## Overview

Naive Bayes comes in three variants: Gaussian for continuous features, multinomial for counts, and Bernoulli for binary indicators. The right pick is dictated by the **shape of your features**, not the domain. **Prerequisites:** [1-introduction.md](1-introduction.md) and the math from [2-math-foundation.md](2-math-foundation.md).

### Choosing the Right Type

Think of choosing a Naive Bayes type like choosing the right tool for a job:

- Need to measure something? Use a ruler (Gaussian NB)
- Counting things? Use a tally counter (Multinomial NB)
- Checking if something is present? Use a checklist (Bernoulli NB)

{% include mermaid-diagram.html src="5-ml-fundamentals/5.2-supervised-learning-1/naive-bayes/diagrams/3-types-1.mmd" %}

## 1. Gaussian Naive Bayes: For Numbers

![Gaussian NB: overlapping class distributions with decision boundary](assets/gaussian_nb.png)

### What is it?

Gaussian Naive Bayes is like a smart ruler that understands how numbers are distributed. It's perfect for:

- Height and weight measurements
- Temperature readings
- Age data
- Any continuous numbers

### Real-World Example: Medical Diagnosis

Imagine you're a doctor trying to predict if a patient has a certain disease based on their:

- Body temperature
- Heart rate
- Blood pressure
- Age

These are all numbers, so Gaussian NB is perfect!

#### Gaussian NB on scaled vitals

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# Example: Predicting disease based on vital signs
patient_data = [
    [38.5, 90, 140, 45],  # [temperature, heart_rate, blood_pressure, age]
    [37.0, 70, 120, 30],
    [39.0, 95, 150, 55]
]
diagnoses = ['sick', 'healthy', 'sick']

# Always scale your numbers!
scaler = StandardScaler()
scaled_data = scaler.fit_transform(patient_data)

# Create and train the model
model = GaussianNB()
model.fit(scaled_data, diagnoses)

# Predict for a new patient
new_patient = [[38.2, 85, 135, 40]]
scaled_new_patient = scaler.transform(new_patient)
prediction = model.predict(scaled_new_patient)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-12" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Data and Scaling</span>
    </div>
    <div class="code-callout__body">
      <p>Three patient records with four vitals are labelled sick/healthy; StandardScaler normalises the features so that temperature (38°C) and blood pressure (140 mmHg) live on comparable scales before Gaussian NB estimates the per-class distributions.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="14-22" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Fit and Predict</span>
    </div>
    <div class="code-callout__body">
      <p>GaussianNB learns mean and variance for each feature per class; the new patient is transformed with the same scaler fitted on training data before calling <code>predict</code>.</p>
    </div>
  </div>
</aside>
</div>

### Why This Matters

Gaussian NB is great because:

- It understands how numbers are distributed
- Works well with measurements
- Can handle different scales (like temperature and age)
- Fast and efficient

## 2. Multinomial Naive Bayes: For Counting

![Word frequencies per class: spam words like 'buy' and 'free' vs ham words like 'meeting' and 'agenda'](assets/multinomial_feature_counts.png)

### What is it?

Multinomial Naive Bayes is like a word counter that helps classify text. It's perfect for:

- Document classification
- Spam detection
- Sentiment analysis
- Any data where you're counting things

### Real-World Example: News Article Classification

Imagine you're building a system to automatically categorize news articles into:

- Sports
- Politics
- Technology
- Entertainment

Multinomial NB counts how often words appear in each category to make its predictions.

#### Multinomial NB on word counts

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Example: Categorizing news articles
articles = [
    "The team won the championship last night",
    "New technology breakthrough in AI research",
    "Political debate scheduled for next week"
]
categories = ['sports', 'tech', 'politics']

# Convert text to word counts
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(articles)

# Create and train the model
model = MultinomialNB()
model.fit(X, categories)

# Predict a new article
new_article = ["The new smartphone features amazing camera technology"]
X_new = vectorizer.transform(new_article)
prediction = model.predict(X_new)  # Should predict 'tech'
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-11" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Corpus and Vectorise</span>
    </div>
    <div class="code-callout__body">
      <p>Three one-sentence training articles cover sports, tech, and politics; <code>CountVectorizer().fit_transform</code> builds a sparse document-term matrix of raw word counts used as features.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-22" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Train and Classify</span>
    </div>
    <div class="code-callout__body">
      <p>MultinomialNB models word counts as multinomial distributions per class; the new article is transformed with the same vocabulary before predict, words like "smartphone" and "technology" should push it toward "tech".</p>
    </div>
  </div>
</aside>
</div>

### Why This Matters

Multinomial NB is great because:

- Perfect for text data
- Handles word frequencies well
- Works with any kind of count data
- Very efficient with large datasets

## 3. Bernoulli Naive Bayes: For Yes/No Questions

### What is it?

Bernoulli Naive Bayes is like a checklist that only cares if something is present or not. It's perfect for:

- Binary features (yes/no)
- Presence/absence data
- Features that are either true or false

### Real-World Example: Email Spam Detection

Imagine you're building a spam filter that checks for:

- Contains the word "free"? (yes/no)
- Has attachments? (yes/no)
- Contains links? (yes/no)
- Has exclamation marks? (yes/no)

Bernoulli NB is perfect for these yes/no features!

#### Bernoulli NB on binary feature vectors

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
from sklearn.naive_bayes import BernoulliNB

# Example: Spam detection
email_features = [
    [1, 1, 0, 1],  # [has_free, has_attachment, has_link, has_exclamation]
    [0, 0, 1, 0],
    [1, 1, 0, 1]
]
labels = ['spam', 'not_spam', 'spam']

# Create and train the model
model = BernoulliNB()
model.fit(email_features, labels)

# Predict a new email
new_email = [[1, 0, 0, 1]]  # has_free=True, has_exclamation=True
prediction = model.predict(new_email)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-9" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Binary Features</span>
    </div>
    <div class="code-callout__body">
      <p>Each email is encoded as four binary flags (has_free, has_attachment, has_link, has_exclamation); BernoulliNB models each feature as a Bernoulli trial (present/absent) rather than a word count.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="11-17" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Fit and Predict</span>
    </div>
    <div class="code-callout__body">
      <p>The model learns P(feature=1 | class) for each binary flag; a new email with has_free and has_exclamation both true maps to the spam-like pattern seen in training.</p>
    </div>
  </div>
</aside>
</div>

### Why This Matters

Bernoulli NB is great because:

- Simple and fast
- Perfect for binary features
- Works well with presence/absence data
- Less sensitive to word frequency than Multinomial NB

## Choosing the Right Type: A Quick Guide

### Decision Tree

{% include mermaid-diagram.html src="5-ml-fundamentals/5.2-supervised-learning-1/naive-bayes/diagrams/3-types-2.mmd" %}

### Quick Reference Table

| Type | Best For | Example | When to Use |
|------|----------|---------|-------------|
| Gaussian | Numbers | Height, Temperature | When dealing with measurements |
| Multinomial | Counts | Word frequencies | When counting occurrences |
| Bernoulli | Yes/No | Feature presence | When only presence matters |

## Common Mistakes to Avoid

1. **Using the Wrong Type**
   - Don't use Gaussian for text data
   - Don't use Multinomial for binary features
   - Don't use Bernoulli for continuous numbers

2. **Forgetting to Preprocess**
   - Scale numbers for Gaussian NB
   - Convert text to counts for Multinomial NB
   - Ensure binary features for Bernoulli NB

3. **Ignoring Data Characteristics**
   - Check if your data matches the type's assumptions
   - Transform data if needed
   - Consider mixing types for different features

## Practice Time

Try these exercises:

1. Build a spam detector using Bernoulli NB
2. Create a document classifier with Multinomial NB
3. Predict medical conditions using Gaussian NB
4. Compare the performance of different types

## Gotchas

- **Passing negative values to `MultinomialNB`**: `MultinomialNB` assumes features are non-negative counts. If you pass TF-IDF scores (which are floats, not integers) or any negative values, scikit-learn will raise a `ValueError`. Use `BernoulliNB` for binary features, `GaussianNB` for real-valued features, and `MultinomialNB` only when your features are genuine non-negative integers or TF-IDF with `min_df` safely applied.
- **Using `GaussianNB` without scaling and then wondering why accuracy is low**: Unlike many parametric models, `GaussianNB` estimates per-class Gaussian distributions. Unscaled features (e.g., blood pressure in the 100s vs age in the 30s) don't cause the model to fail outright, but the variance estimates become dominated by the high-magnitude feature, skewing the decision boundary. Always scale continuous inputs.
- **Applying `BernoulliNB` to word-count features instead of binary presence**: `BernoulliNB` models each feature as present-or-absent (0 or 1). If you feed raw word counts (e.g., "free" appears 3 times), it treats anything non-zero as 1, silently discarding frequency information. Use `MultinomialNB` when word frequency matters, and `BernoulliNB` only when you want pure presence/absence.
- **Calling `scaler.fit_transform` on the new patient at prediction time**: The `GaussianNB` example calls `scaler.transform(new_patient)` (correct). A common mistake is to call `scaler.fit_transform(new_patient)`, which fits a new scaler on a single row of data, producing useless z-scores and a silently wrong prediction.
- **Expecting `MultinomialNB` to handle out-of-vocabulary words gracefully without smoothing**: The `CountVectorizer` ignores words in test documents that didn't appear in training. This is handled automatically by sklearn's pipeline, but if you build your own vocabulary manually and forget to handle OOV tokens, the model will either error or produce zero-probability predictions for those documents.

## Next Steps

Ready to put these types into action? Move on to [Implementation](4-implementation.md) to see how to use these different types in real projects.

Remember: The right tool for the right job! Choose your Naive Bayes type wisely based on your data.
