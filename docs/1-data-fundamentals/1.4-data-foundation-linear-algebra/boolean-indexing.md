# Boolean Indexing: Smart Data Selection

**After this lesson:** you can explain the core ideas in “Boolean Indexing: Smart Data Selection” and reproduce the examples here in your own notebook or environment.

### Video

<div class="video-embed">
<iframe width="560" height="315" src="https://www.youtube.com/embed/QUT1VHiLmmI" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

*freeCodeCamp — Python NumPy tutorial for beginners*

## Overview

**Prerequisites:** [Introduction to NumPy](./intro-numpy.md) and [ndarray basics](./ndarray-basic.md) (creating arrays and slicing).

**Why this lesson:** **Boolean indexing** selects elements with a mask of `True`/`False` the same shape as your data. It is how you express “keep rows where score ≥ 80” without slow Python loops—essential before pandas boolean filters.

## What is Boolean Indexing?

Think of boolean indexing as a smart filter for your data - like having a magic sieve that only lets through the items you want! It's one of NumPy's most powerful features for data analysis, letting you:

- Filter data based on conditions
- Clean data by removing unwanted values
- Analyze specific subsets of your data
- Replace values that meet certain criteria

Real-world applications:

- Finding stocks above a certain price
- Identifying temperatures above freezing
- Filtering out invalid measurements
- Finding transactions over a certain amount

---

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

```
Students: ['Bob' 'Joe' 'Will' 'Bob' 'Will' 'Joe' 'Joe']

Scores:
[[ 75  80]
 [ 85  90]
 [ 95 100]
 [100  77]
 [ 85  92]
 [ 95  80]
 [ 72  80]]

Average scores:
Math: 86.71428571428571
Science: 85.57142857142857
```

Real-world scenario - Finding high performers:

```python
# Create boolean mask for high achievers (90+ in either subject)
high_scores = (scores >= 90).any(axis=1)
print("\nHigh achievers:")
print("Names:", names[high_scores])
print("Their scores:\n", scores[high_scores])
```

```

High achievers:
Names: ['Joe' 'Will' 'Bob' 'Will' 'Joe']
Their scores:
 [[ 85  90]
 [ 95 100]
 [100  77]
 [ 85  92]
 [ 95  80]]
```

---

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

```
Bob mask: [ True False False  True False False False]

Bob's scores:
[[ 75  80]
 [100  77]]

Bob's averages:
Math: 87.5
Science: 78.5
Bob's best subject: Math
```

## Cool Filtering Tricks

---

### Not Bob (Using ~)

```python
# Get everyone except Bob
not_bob = ~(names == "Bob")
print(scores[not_bob])

# Same thing using !=
also_not_bob = (names != "Bob")
print(scores[also_not_bob])
```

```
[[ 85  90]
 [ 95 100]
 [ 85  92]
 [ 95  80]
 [ 72  80]]
[[ 85  90]
 [ 95 100]
 [ 85  92]
 [ 95  80]
 [ 72  80]]
```

---

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

```
[[ 75  80]
 [ 95 100]
 [100  77]
 [ 85  92]]
High scores mask:
[[False False]
 [ True  True]
 [ True  True]
 [ True False]
 [ True  True]
 [ True False]
 [False False]]
```

---

### Changing Values with Masks

```python
# Set all scores below 80 to 70
scores[scores < 80] = 70

# Before:     After:
# 75 → 70     72 → 70
# 77 → 70     All others unchanged
```

## Visual Guide to Boolean Indexing

---

### How Masks Work

```
Names:  ["Bob", "Joe", "Will", "Bob"]
Mask:   [True, False, False, True]
        ↓      ↓      ↓      ↓
Result: [Bob's data,      Bob's data]
```

---

### Combining Conditions

```
Condition 1:  [True,  False, True,  False]
     AND (&)  
Condition 2:  [True,  True,  False, False]
     =
    Result:   [True,  False, False, False]
```

**Pro Tips**:

- Use **==** for exact matches
- Use **~** to invert a condition
- Use **|** for OR, **&** for AND
- Conditions can be combined with parentheses
- Think of masks as "keeping" (**True**) or "filtering out" (**False**)

## Common pitfalls

- **Chaining comparisons** — Write **(a < x) & (x < b)**; Python’s chained comparisons do not broadcast over arrays the way you might expect in all cases.
- **Precedence** — **&** binds tighter than you expect; wrap each condition in parentheses.
- **Non-boolean dtypes** — Masks must be boolean; compare with **==**, **<**, etc., not raw floats meant as probabilities unless you threshold.

## Next steps

Continue to [ndarray methods](./ndarray-methods.md), then [Linear algebra](./linear-algebra.md).
