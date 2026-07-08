---
reading_minutes: 20
objectives:
  - "Distinguish a decision tree's root, decision, and leaf nodes from the if/else rules they encode."
  - "Fit a `DecisionTreeClassifier` to a small tabular dataset and read `feature_importances_` to see which features drove the splits."
  - "Recognise where decision trees shine (interpretable rules, mixed feature types) and where they struggle (smooth boundaries, very high dimensions)."
---
# Introduction to Decision Trees

**After this lesson:** you can explain Introduction to Decision Trees and try the examples in your own notebook.

## Overview

A **decision tree** splits the data with nested if/else rules on features until it reaches a prediction, easy to visualize and explain. This page uses a tiny "go for a walk" example; follow with [tree structure](2-tree-structure.md) and [implementation](3-implementation.md). **Prerequisites:** [Supervised learning 5.2](../README.md); sklearn `fit` / `predict` from [5.1 workflow](../../5.1-intro-to-ml/ml-workflow.md).


## Why Learn Decision Trees?

Decision trees are one of the most intuitive and powerful machine learning algorithms. They're perfect for beginners because:

- They mimic how humans make decisions
- They're easy to visualize and understand
- They can handle both numbers and categories
- They provide clear explanations for predictions

![Basic Tree Structure](assets/basic_tree_structure.png)

## What is a Decision Tree?

Imagine you're trying to decide whether to go for a walk. You might ask yourself:

1. Is it raining? (No)
2. Is it too hot? (No)
3. Do I have time? (Yes)
4. Then I'll go for a walk!

This is exactly how a decision tree works! It's a series of yes/no questions that lead to a final decision.

### Real-World Examples

1. **Medical Diagnosis**
   - Doctor: "Do you have a fever?" (Yes)
   - Doctor: "Is it above 101°F?" (No)
   - Doctor: "Do you have a cough?" (Yes)
   - Diagnosis: "You might have a mild infection"

2. **Loan Approval**
   - Bank: "Is your income above $50,000?" (Yes)
   - Bank: "Is your credit score above 700?" (No)
   - Bank: "Do you have any existing loans?" (No)
   - Decision: "Loan approved with higher interest rate"

## Key Components of a Decision Tree

Break down the parts of a decision tree using a simple example:

#### Train a toy tree and inspect structure

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Create a simple dataset for the "go for a walk" decision
# Features: [is_raining, temperature, have_time]
# Values: 0 = No, 1 = Yes for is_raining and have_time
# Temperature is in Fahrenheit
X = np.array([
    [0, 72, 1],  # Not raining, 72°F, have time
    [1, 68, 1],  # Raining, 68°F, have time
    [0, 95, 1],  # Not raining, 95°F, have time
    [0, 75, 0],  # Not raining, 75°F, no time
    [1, 70, 0],  # Raining, 70°F, no time
    [0, 65, 1],  # Not raining, 65°F, have time
    [1, 78, 1],  # Raining, 78°F, have time
    [0, 82, 1]   # Not raining, 82°F, have time
])

# Decision: 1 = go for walk, 0 = don't go for walk
y = np.array([1, 0, 0, 0, 0, 1, 0, 1])

# Create and train the model
tree_model = DecisionTreeClassifier(max_depth=3)
tree_model.fit(X, y)

# Visualize the tree
plt.figure(figsize=(12, 8))
plot_tree(
    tree_model,
    feature_names=['Is Raining', 'Temperature', 'Have Time'],
    class_names=['Stay Home', 'Go for Walk'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title('Decision Tree for Walking Decision')
plt.show()

# Make a prediction for a new scenario
new_scenario = np.array([[0, 70, 1]])  # Not raining, 70°F, have time
prediction = tree_model.predict(new_scenario)
print(f"Decision: {'Go for a walk' if prediction[0] == 1 else 'Stay home'}")

# Explain the prediction
feature_importance = tree_model.feature_importances_
features = ['Is Raining', 'Temperature', 'Have Time']
for i, importance in enumerate(feature_importance):
    print(f"{features[i]}: {importance:.2f}")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-25" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Data and Tree Fit</span>
    </div>
    <div class="code-callout__body">
      <p>Eight hand-crafted walking scenarios with three features (rain, temperature, time) are fitted by <code>DecisionTreeClassifier(max_depth=3)</code>; the binary label encodes the go/stay decision.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="27-40" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Visualize Tree</span>
    </div>
    <div class="code-callout__body">
      <p><code>plot_tree</code> with <code>filled=True</code> colors each node by majority class, making it easy to trace any path from root to leaf and understand the splitting logic.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="42-51" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Predict and Explain</span>
    </div>
    <div class="code-callout__body">
      <p>Predict for a new scenario, then print <code>feature_importances_</code>, each value is the fraction of total impurity reduction attributed to that feature across all splits.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/1-introduction_fig_1.png" alt="1-introduction" />
<figcaption>Figure 1: Decision Tree for Walking Decision</figcaption>
</figure>

```
Decision: Go for a walk
Is Raining: 0.49
Temperature: 0.51
Have Time: 0.00
```

In this decision tree:

1. **Root Node**: The starting point at the top, representing the entire dataset.
2. **Decision Nodes**: Places where the tree splits based on a feature question.
3. **Leaf Nodes**: The end points where we make predictions.
4. **Branches**: Connections between nodes, representing answers to questions.

### Key Terms to Understand

1. **Features** (or attributes): The characteristics we use to make decisions.
   - In our walking example: weather, temperature, available time

2. **Splitting Criteria**: How we decide which feature to use for splitting.
   - Common metrics: Gini impurity, information gain, entropy

3. **Impurity**: How mixed the classes are in a node.
   - Pure node: All samples belong to one class
   - Impure node: Mix of different classes

4. **Pruning**: Removing unnecessary branches to prevent overfitting.
   - Like editing a story to remove unnecessary details

## How Decision Trees Learn

Decision trees learn by finding the best questions to ask that separate the data most effectively:

#### 2D synthetic data: decision surface + tree diagram

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
# Look at the splitting process visually
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Create a simple 2D dataset
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1
)

# Create and train the model
tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(X, y)

# Get the decision boundary
def plot_decision_boundary(model, X, y):
    # Set min and max values for plotting
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predict for each point in the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolor='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Tree Boundary')
    plt.show()

# Plot the decision boundary
plot_decision_boundary(tree_clf, X, y)

# Plot the actual tree
plt.figure(figsize=(12, 8))
plot_tree(
    tree_clf,
    feature_names=['Feature 1', 'Feature 2'],
    class_names=['Class 0', 'Class 1'],
    filled=True,
    rounded=True
)
plt.title('Decision Tree Structure')
plt.show()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-19" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Generate and Fit</span>
    </div>
    <div class="code-callout__body">
      <p><code>make_classification</code> with <code>n_features=2</code> produces a 2D dataset ideal for visualizing decision boundaries; <code>max_depth=3</code> keeps the tree readable.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="21-43" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Decision Boundary Helper</span>
    </div>
    <div class="code-callout__body">
      <p>The helper builds a fine meshgrid, predicts class labels at each point, and fills contour regions, this reveals the axis-aligned rectangular regions that decision trees always produce.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="45-57" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Boundary and Tree Plots</span>
    </div>
    <div class="code-callout__body">
      <p>Plot the decision surface first, then <code>plot_tree</code> to see the exact splits that produced those rectangular regions side by side.</p>
    </div>
  </div>
</aside>
</div>


<figure>
<img src="assets/1-introduction_fig_2.png" alt="1-introduction" />
<figcaption>Figure 2: Decision Tree Boundary</figcaption>
</figure>


<figure>
<img src="assets/1-introduction_fig_3.png" alt="1-introduction" />
<figcaption>Figure 3: Decision Tree Structure</figcaption>
</figure>



The tree learning process:

1. **Start** with all data in the root node
2. **Find** the best feature and threshold to split the data
3. **Create** child nodes based on the split
4. **Repeat** steps 2-3 for each child node
5. **Stop** when a stopping condition is met (e.g., maximum depth)

## Advantages of Decision Trees

1. **Easy to understand**: They mimic human decision-making
2. **No data preprocessing needed**: No scaling or normalization required
3. **Handle mixed data types**: Both categorical and numerical features
4. **Non-linear relationships**: Can capture complex patterns
5. **Feature importance**: Automatically identify important features

## Limitations of Decision Trees

1. **Overfitting**: Tend to create overly complex trees that don't generalize well
2. **Instability**: Small changes in data can result in very different trees
3. **Biased toward features with more levels**: Can favor features with many unique values
4. **Can't extrapolate**: Can only make predictions within the range of training data

## When to Use Decision Trees

Decision trees are ideal for:

1. **Classification problems** with categorical or numerical features
2. **Interpretable models** where understanding the decision process is important
3. **Feature selection** to identify important variables
4. **As components in more powerful ensemble methods** (Random Forests, Gradient Boosting)

## Practice Exercise

Try building a simple decision tree on your own:

#### Exercise: student pass/fail (`5.2-dt-1-intro-exercise`)

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
# Simple exercise: Predicting if a student will pass or fail
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Features: [hours_studied, previous_score, sleep_hours, attendance_percentage]
X_students = np.array([
    [8, 85, 7, 90],   # Student 1
    [3, 70, 5, 75],   # Student 2
    [5, 77, 6, 85],   # Student 3
    [2, 65, 4, 70],   # Student 4
    [7, 90, 8, 95],   # Student 5
    [4, 72, 6, 80],   # Student 6
    [6, 81, 7, 88],   # Student 7
    [3, 68, 5, 65]    # Student 8
])

# Result: 1 = pass, 0 = fail
y_results = np.array([1, 0, 1, 0, 1, 0, 1, 0])

# Create and train model
student_model = DecisionTreeClassifier()
student_model.fit(X_students, y_results)

# Make a prediction for a new student
new_student = np.array([[6, 75, 7, 82]])
result = student_model.predict(new_student)
print(f"Prediction: {'Pass' if result[0] == 1 else 'Fail'}")

# Get feature importance
features = ['Hours Studied', 'Previous Score', 'Sleep Hours', 'Attendance']
importances = student_model.feature_importances_

# Display the most important factor in passing
most_important = features[np.argmax(importances)]
print(f"Most important factor: {most_important}")
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-20" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Student Dataset</span>
    </div>
    <div class="code-callout__body">
      <p>Eight students with four study-habit features and a binary pass/fail label; the data is intentionally small so the tree structure is easy to inspect manually.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="21-37" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Predict and Rank Features</span>
    </div>
    <div class="code-callout__body">
      <p>After fitting, <code>predict</code> classifies a new student; <code>np.argmax(feature_importances_)</code> identifies the single most influential feature across all splits in the learned tree.</p>
    </div>
  </div>
</aside>
</div>

```
Prediction: Pass
Most important factor: Previous Score
```

## Gotchas

- **Fitting a `DecisionTreeClassifier` without `max_depth` on tiny toy data**: the unrestrained tree in the student exercise will memorise the 8 training rows perfectly (training accuracy 1.0) but will often fail on any new student; always set a depth limit or use cross-validation to see through the perfect training score.
- **Misreading `feature_importances_` as a global ranking**: importance values are specific to this fitted tree on this training set; a feature with zero importance was not useful given the *other* features present, not necessarily unimportant in general; run the same data with a different split and the ranking can change.
- **Assuming the "Go for a Walk" labels are deterministic**: the training data has only 8 rows with mixed outcomes for similar conditions; the learned tree may predict confidently for a new scenario simply because it memorised an adjacent training row, not because it found a true pattern.
- **Calling `plot_tree` without naming features and classes**: the default output shows numeric feature indices and class integers, which are nearly unreadable; always pass `feature_names` and `class_names` before sharing or debugging a tree.
- **Treating `feature_importances_` totalling 1.0 as a percentage of predictive power**: importance measures impurity reduction, not correlation with the target; a single dominant feature (e.g., "Temperature: 0.51") does not mean the other features are useless, only that they contributed less to this tree's splits.

## Next Steps

Now that you understand the basics of decision trees, we will look at:

1. [How decision trees are structured](2-tree-structure.md)
2. [How to implement them in Python](3-implementation.md)
3. [Advanced techniques and optimizations](4-advanced.md)
4. [Real-world applications](5-applications.md)
