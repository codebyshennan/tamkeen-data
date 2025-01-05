# Boolean Indexing: Smart Data Selection! 🎯

## What is Boolean Indexing? 🤔

Think of boolean indexing as a smart filter for your data - like having a magic sieve that only lets through the items you want! It's one of NumPy's most powerful features for data analysis, letting you:

- 🔍 Filter data based on conditions
- ✨ Clean data by removing unwanted values
- 📊 Analyze specific subsets of your data
- 🔄 Replace values that meet certain criteria

Real-world applications:
- 📈 Finding stocks above a certain price
- 🌡️ Identifying temperatures above freezing
- 📊 Filtering out invalid measurements
- 💰 Finding transactions over a certain amount

{% stepper %}
{% step %}
### Setup: Student Scores Example
```python
import numpy as np

# Student names (some repeated)
names = np.array(["Bob", "Joe", "Will", "Bob", "Will", "Joe", "Joe"])

# Their test scores [math, science]
scores = np.array([
    [75, 80],  # Bob's scores
    [85, 90],  # Joe's scores
    [95, 100], # Will's scores
    [100, 77], # Bob's scores
    [85, 92],  # Will's scores
    [95, 80],  # Joe's scores
    [72, 80]   # Joe's scores
])

print("Students:", names)
print("\nScores:")
print(scores)

# Basic statistics
print("\nAverage scores:")
print("Math:", scores[:, 0].mean())    # First column (math)
print("Science:", scores[:, 1].mean())  # Second column (science)
```

Real-world scenario - Finding high performers:
```python
# Create boolean mask for high achievers (90+ in either subject)
high_scores = (scores >= 90).any(axis=1)
print("\nHigh achievers:")
print("Names:", names[high_scores])
print("Their scores:\n", scores[high_scores])
```
{% endstep %}

{% step %}
### Finding Students
```python
# Create a mask for "Bob"
bob_mask = names == "Bob"
print("Bob mask:", bob_mask)  # [True False False True False False False]

# Get Bob's scores
bob_scores = scores[bob_mask]
print("\nBob's scores:")
print(bob_scores)

# Calculate Bob's averages
print("\nBob's averages:")
print("Math:", bob_scores[:, 0].mean())
print("Science:", bob_scores[:, 1].mean())

# Find Bob's best subject
subjects = ['Math', 'Science']
best_subject = subjects[bob_scores.mean(axis=0).argmax()]
print(f"Bob's best subject: {best_subject}")
```
{% endstep %}
{% endstepper %}

## Cool Filtering Tricks! 🎨

{% stepper %}
{% step %}
### Not Bob (Using ~)
```python
# Get everyone except Bob
not_bob = ~(names == "Bob")
print(scores[not_bob])

# Same thing using !=
also_not_bob = (names != "Bob")
print(scores[also_not_bob])
```
{% endstep %}

{% step %}
### Multiple Conditions
```python
# Get Bob OR Will
bob_or_will = (names == "Bob") | (names == "Will")
print(scores[bob_or_will])

# Get high scores (> 80)
high_scores = scores > 80
print("High scores mask:")
print(high_scores)
```
{% endstep %}

{% step %}
### Changing Values with Masks
```python
# Set all scores below 80 to 70
scores[scores < 80] = 70

# Before:     After:
# 75 → 70     72 → 70
# 77 → 70     All others unchanged
```
{% endstep %}
{% endstepper %}

## Visual Guide to Boolean Indexing 🎨

{% stepper %}
{% step %}
### How Masks Work
```
Names:  ["Bob", "Joe", "Will", "Bob"]
Mask:   [True, False, False, True]
        ↓      ↓      ↓      ↓
Result: [Bob's data,      Bob's data]
```
{% endstep %}

{% step %}
### Combining Conditions
```
Condition 1:  [True,  False, True,  False]
     AND (&)  
Condition 2:  [True,  True,  False, False]
     =
    Result:   [True,  False, False, False]
```
{% endstep %}
{% endstepper %}

💡 **Pro Tips**:
- Use `==` for exact matches
- Use `~` to invert a condition
- Use `|` for OR, `&` for AND
- Conditions can be combined with parentheses
- Think of masks as "keeping" (`True`) or "filtering out" (`False`)
