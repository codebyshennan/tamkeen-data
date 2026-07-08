# NumPy Array Methods: Shape-Shifting Magic

**After this lesson:** you can explain NumPy Array Methods: Shape-Shifting Magic and try the examples in your own notebook.

### Video

<div class="video-embed">
<iframe width="560" height="315" src="https://www.youtube.com/embed/kYB8IZa5AuE" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

*3Blue1Brown, Linear transformations and matrices (Essence of linear algebra)*

## Overview

**Prerequisites:** [Introduction to NumPy](./intro-numpy.md), [ndarray](./ndarray.md) concept, and [basic array creation and math](./ndarray-basic.md).

**Why this lesson:** Real pipelines **reshape**, **stack**, **split**, and **aggregate** arrays to match model inputs and plot layouts. Knowing `reshape`, `axis`, and reductions (`sum`, `mean`) prevents silent shape bugs.

## Reshaping Arrays: The Transformer

---

### What is Reshaping?

Think of reshaping like rearranging chairs in a room - same number of chairs, different arrangement! It's useful when you need to:

{% include mermaid-diagram.html src="1-data-fundamentals/1.4-data-foundation-linear-algebra/diagrams/ndarray-methods-1.mmd" %}

*`-1` in `reshape` means "infer this dimension". `reshape(1,-1)` is the most common fix when sklearn complains about a 1D input that needs to be 2D.*

- Convert 1D data into a 2D table format
- Restructure data to match another array's shape
- Transform image data (e.g., flattening for ML models)
- Prepare data for plotting or analysis

Key concepts:

1. Total elements must stay the same
2. Order of elements is preserved
3. Shape is specified as (rows, columns)

```python
import numpy as np

# Create array with numbers 0-14
arr = np.arange(15)
print("Original:", arr)
print("Shape:", arr.shape)  # (15,)

# Reshape to 3 rows, 5 columns
matrix = arr.reshape((3, 5))
print("\nReshaped to 3x5:")
print(matrix)
print("New shape:", matrix.shape)  # (3, 5)

# Reshape to 5 rows, 3 columns
matrix2 = arr.reshape((5, 3))
print("\nReshaped to 5x3:")
print(matrix2)
print("New shape:", matrix2.shape)  # (5, 3)

# Real-world example: Image processing
# Simulate RGB image data (3 channels)
image_data = np.random.randint(0, 256, size=(4, 4, 3))  # 4x4 RGB image
print("\nOriginal image shape:", image_data.shape)  # (4, 4, 3)

# Flatten for ML model
flattened = image_data.reshape(-1)  # -1 means "figure out this dimension"
print("Flattened shape:", flattened.shape)  # (48,)
```

```
Original: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]
Shape: (15,)

Reshaped to 3x5:
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
New shape: (3, 5)

Reshaped to 5x3:
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]
 [12 13 14]]
New shape: (5, 3)

Original image shape: (4, 4, 3)
Flattened shape: (48,)
```

---

### Visual Guide to Reshaping

```
Before (1D):
[0 1 2 3 4 5 6 7 8 9 10 11]

After reshape(3,4):         After reshape(4,3):
┌─────────────┐            ┌───────┐
│ 0  1  2  3  │            │ 0 1 2 │
│ 4  5  6  7  │            │ 3 4 5 │
│ 8  9  10 11 │            │ 6 7 8 │
└─────────────┘            │ 9 10 11│
                          └───────┘

Common shapes:
- (n,)     → 1D array with n elements
- (n,1)    → Column vector
- (1,n)    → Row vector
- (m,n)    → m×n matrix
- (h,w,c)  → Image with height h, width w, c channels
```

## Transposing: The Flip Master

---

### What is Transposing?

Transposing is like looking at your data from the side - rows become columns and columns become rows!

```python
# Original array
arr = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Transpose it
print("Transposed:")
print(arr.T)  # or arr.transpose()
# [[1 4]
#  [2 5]
#  [3 6]]
```

```
Transposed:
[[1 4]
 [2 5]
 [3 6]]
```

---

### Visual Guide to Transposing

```
Before:        After:
┌─────────┐    ┌─────┐
│ 1 2 3   │    │ 1 4 │
│ 4 5 6   │ => │ 2 5 │
└─────────┘    │ 3 6 │
               └─────┘
```

## Universal Functions: Math Wizardry! ✨

---

### What are Universal Functions?

They're like magic spells that work on every element in your array at once!

```python
# Create array 0-9
arr = np.arange(10)
print("Original:", arr)

# Square root of everything!
print("Square roots:", np.sqrt(arr))

# e raised to each power
print("Exponentials:", np.exp(arr))
```

```
Original: [0 1 2 3 4 5 6 7 8 9]
Square roots: [0.         1.         1.41421356 1.73205081 2.         2.23606798
 2.44948974 2.64575131 2.82842712 3.        ]
Exponentials: [1.00000000e+00 2.71828183e+00 7.38905610e+00 2.00855369e+01
 5.45981500e+01 1.48413159e+02 4.03428793e+02 1.09663316e+03
 2.98095799e+03 8.10308393e+03]
```

---

### One Array vs Two Arrays

```python
# One array operations (Unary)
x = np.array([1, 4, 9])
print("Square roots:", np.sqrt(x))  # [1, 2, 3]

# Two array operations (Binary)
a = np.array([3, 7, 15, 5, 12])
b = np.array([11, 2, 4, 6, 8])
print("Maximum values:", np.maximum(a, b))
```

```
Square roots: [1. 2. 3.]
Maximum values: [11  7 15  6 12]
```

---

### Visual Guide to Universal Functions

```
One Array (sqrt):
Input:  [1  4  9]
         ↓  ↓  ↓  √
Output: [1  2  3]

Two Arrays (maximum):
Array1: [3  7  15]
Array2: [11 2  4 ]
         ↓  ↓  ↓  max
Output: [11 7  15]
```

## Smart choices with `where()`

---

### What is where()?

Think of it as a smart chooser - "If this, pick that, otherwise pick this"

```python
# Set up our choices
x = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
y = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
conditions = np.array([True, False, True, True, False])

# Choose based on conditions
result = np.where(conditions, x, y)
print("Result:", result)
# [1.1 2.2 1.3 1.4 2.5]
```

```
Result: [1.1 2.2 1.3 1.4 2.5]
```

---

### Visual Guide to where()

```
Condition: [True  False True  True  False]
X values:  [1.1   1.2   1.3   1.4   1.5 ]
Y values:  [2.1   2.2   2.3   2.4   2.5 ]
           ↓      ↓      ↓      ↓      ↓
Result:    [1.1   2.2   1.3   1.4   2.5 ]
           (X)    (Y)    (X)    (X)    (Y)
```

## Array Statistics: Number Crunching

---

### Basic Statistics

```python
# Create a random array
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print("Mean:", arr.mean())
print("Sum:", arr.sum())
print("Min:", arr.min())
print("Max:", arr.max())
```

```
Mean: 5.0
Sum: 45
Min: 1
Max: 9
```

---

### Computing Along Axes

```python
# Mean of each row (axis=1)
print("Row means:", arr.mean(axis=1))

# Mean of each column (axis=0)
print("Column means:", arr.mean(axis=0))
```

```
Row means: [2. 5. 8.]
Column means: [4. 5. 6.]
```

Visual guide to axes:

```
axis=0 (down columns)    axis=1 (across rows)
    ↓   ↓   ↓             →→→
┌───────────┐           ┌───────────┐
│ 1  2  3   │ →         │ 1  2  3   │
│ 4  5  6   │ →         │ 4  5  6   │
│ 7  8  9   │ →         │ 7  8  9   │
└───────────┘           └───────────┘
```

## Boolean Operations: Truth Seekers

---

### Testing Arrays

```python
# Create boolean array
bools = np.array([False, False, True, False])

# Check if any are True
print("Any True?", bools.any())  # True

# Check if all are True
print("All True?", bools.all())  # False
```

```
Any True? True
All True? False
```

---

### Sorting Arrays

```python
# Create random array
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print("Before:", arr)

# Sort in place
arr.sort()
print("After:", arr)
```

```
Before: [3 1 4 1 5 9 2 6]
After: [1 1 2 3 4 5 6 9]
```

**Pro Tips**:

- Use **reshape** when you need to change array dimensions
- Remember: rows → columns with **transpose** or **.T**
- Universal functions are super fast, use them
- **where** is great for conditional operations
- Think about which axis you want when using statistics

## Common pitfalls

- **NaN propagation**: Many reductions return **nan** if any element is **nan**; use **nanmean** and friends when appropriate.
- **keepdims**: Forgetting **keepdims=True** can break broadcasting in the next step.
- **In-place vs return**: Some methods modify the array; others return a new one, check the docs for the function you use.

## Next steps

Continue to [Linear algebra](./linear-algebra.md), then [Data analysis with pandas](../1.5-data-analysis-pandas/README.md) starting with [Series](../1.5-data-analysis-pandas/series.md).
