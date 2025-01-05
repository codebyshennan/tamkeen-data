# NumPy Array Methods: Shape-Shifting Magic! 🔮

## Reshaping Arrays: The Transformer! 🤖

{% stepper %}
{% step %}
### What is Reshaping?
Think of reshaping like rearranging chairs in a room - same number of chairs, different arrangement! It's useful when you need to:

- 📊 Convert 1D data into a 2D table format
- 🔄 Restructure data to match another array's shape
- 📸 Transform image data (e.g., flattening for ML models)
- 📈 Prepare data for plotting or analysis

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
{% endstep %}

{% step %}
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
{% endstep %}
{% endstepper %}

## Transposing: The Flip Master! 🔄

{% stepper %}
{% step %}
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
{% endstep %}

{% step %}
### Visual Guide to Transposing
```
Before:        After:
┌─────────┐    ┌─────┐
│ 1 2 3   │    │ 1 4 │
│ 4 5 6   │ => │ 2 5 │
└─────────┘    │ 3 6 │
               └─────┘
```
{% endstep %}
{% endstepper %}

## Universal Functions: Math Wizardry! ✨

{% stepper %}
{% step %}
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
{% endstep %}

{% step %}
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
{% endstep %}

{% step %}
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
{% endstep %}
{% endstepper %}

## Smart Choices with where()! 🤔

{% stepper %}
{% step %}
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
{% endstep %}

{% step %}
### Visual Guide to where()
```
Condition: [True  False True  True  False]
X values:  [1.1   1.2   1.3   1.4   1.5 ]
Y values:  [2.1   2.2   2.3   2.4   2.5 ]
           ↓      ↓      ↓      ↓      ↓
Result:    [1.1   2.2   1.3   1.4   2.5 ]
           (X)    (Y)    (X)    (X)    (Y)
```
{% endstep %}
{% endstepper %}

## Array Statistics: Number Crunching! 📊

{% stepper %}
{% step %}
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
{% endstep %}

{% step %}
### Computing Along Axes
```python
# Mean of each row (axis=1)
print("Row means:", arr.mean(axis=1))

# Mean of each column (axis=0)
print("Column means:", arr.mean(axis=0))
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
{% endstep %}
{% endstepper %}

## Boolean Operations: Truth Seekers! 🔍

{% stepper %}
{% step %}
### Testing Arrays
```python
# Create boolean array
bools = np.array([False, False, True, False])

# Check if any are True
print("Any True?", bools.any())  # True

# Check if all are True
print("All True?", bools.all())  # False
```
{% endstep %}

{% step %}
### Sorting Arrays
```python
# Create random array
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print("Before:", arr)

# Sort in place
arr.sort()
print("After:", arr)
```
{% endstep %}
{% endstepper %}

💡 **Pro Tips**:
- Use `reshape` when you need to change array dimensions
- Remember: rows → columns with `transpose` or `.T`
- Universal functions are super fast - use them!
- `where` is great for conditional operations
- Think about which axis you want when using statistics
