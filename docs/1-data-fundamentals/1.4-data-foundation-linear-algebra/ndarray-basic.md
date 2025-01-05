# NumPy Array Basics: Math Magic! ✨

## Arithmetic with Arrays 🧮

Ever wished you could do math on entire lists at once? With NumPy arrays, you can! This is called _vectorization_ - it's like having a calculator that works on all numbers simultaneously. Imagine you're:
- 📊 Calculating sales tax on thousands of prices
- 📈 Converting temperatures from Celsius to Fahrenheit
- 💰 Computing compound interest on multiple investments
- 📏 Scaling measurements from inches to centimeters

Instead of writing loops, NumPy lets you perform these operations in one go!

{% stepper %}
{% step %}
### Basic Math Operations
```python
import numpy as np

# Create a 2D array (think of it as a table of numbers)
arr = np.array([
    [1.0, 2.0, 3.0],  # First row
    [4.0, 5.0, 6.0]   # Second row
])

# Basic arithmetic operations
print("Original array:")
print(arr)

print("\nAddition (add 10 to everything):")
print(arr + 10)  # Every number gets 10 added to it

print("\nMultiplication (multiply everything by 2):")
print(arr * 2)   # Every number gets doubled

print("\nPower (square everything):")
print(arr ** 2)  # Every number gets squared

print("\nDivision (divide everything by 2):")
print(arr / 2)   # Every number gets halved

# More complex operations
print("\nSquare root of every number:")
print(np.sqrt(arr))

print("\nExponential (e^x) of every number:")
print(np.exp(arr))
```

Real-world example - Converting temperatures:
```python
# Temperatures in Celsius
celsius = np.array([0, 15, 30, 45])

# Convert to Fahrenheit: F = (C × 9/5) + 32
fahrenheit = (celsius * 9/5) + 32

print("Celsius:", celsius)
print("Fahrenheit:", fahrenheit)
```
{% endstep %}

{% step %}
### How It Works
```
Original Array:     Operation:      Result:
┌─────────────┐     Multiply      ┌─────────────┐
│ 1  2  3     │       ×          │ 1  4  9     │
│ 4  5  6     │     itself       │ 16 25 36    │
└─────────────┘                   └─────────────┘
```
{% endstep %}
{% endstepper %}

![vectorization](./assets/vectorization.png)

## Broadcasting: The Shape-Shifter! 🔄

{% stepper %}
{% step %}
### What is Broadcasting?
It's NumPy's superpower to make arrays of different shapes work together! Think of it as NumPy automatically copying smaller arrays to match bigger ones.

```python
# Original 2D array
arr = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

# Add [1, 1, 1] to each row
print(arr + np.array([1, 1, 1]))
# Result:
# [[2. 3. 4.]
#  [5. 6. 7.]]
```
{% endstep %}

{% step %}
### Magic with Numbers
```python
arr1 = np.array([1, 2, 3, 4])

# Add 4 to everything
print(arr1 + 4)  # [5, 6, 7, 8]

# Square everything
print(arr1 ** 2)  # [1, 4, 9, 16]

# Divide 1 by everything
print(1 / arr1)  # [1.0, 0.5, 0.33, 0.25]
```
{% endstep %}

{% step %}
### How Broadcasting Works
```
Array:     Number:     Result:
[1 2 3]  +    4    =  [1+4 2+4 3+4]
                      [5   6   7  ]

NumPy automatically turns 4 into [4 4 4]!
```
{% endstep %}
{% endstepper %}

## Comparing Arrays 🔍

{% stepper %}
{% step %}
### Array Comparisons
```python
arr2 = np.array([[0.0, 4.0, 1.0],
                 [7.0, 2.0, 12.0]])

# Compare arrays
print(arr2 > arr)
# Result:
# [[False  True False]
#  [ True False  True]]
```
{% endstep %}

{% step %}
### Understanding the Result
```
Array 1:     Compare:    Array 2:     Result:
┌─────────┐     >      ┌─────────┐  ┌─────────┐
│ 1  2  3 │            │ 0  4  1 │  │ F  T  F │
│ 4  5  6 │            │ 7  2  12│  │ T  F  T │
└─────────┘            └─────────┘  └─────────┘
```
{% endstep %}
{% endstepper %}

## Indexing and Slicing: Array Surgery 🔪

{% stepper %}
{% step %}
### Basic Indexing (1D Arrays)
```python
# Create array [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
arr = np.arange(10)

# Get single element
print(arr[5])      # 5

# Get a range
print(arr[5:8])    # [5 6 7]
```
{% endstep %}

{% step %}
### Changing Values
```python
# Change a range to 12
arr[5:8] = 12
print(arr)  # [0 1 2 3 4 12 12 12 8 9]

# Views share memory!
arr_slice = arr[5:8]
arr_slice[1] = 10
print(arr)  # [0 1 2 3 4 12 10 12 8 9]
```
{% endstep %}

{% step %}
### Understanding Slices
```
Index:  0  1  2  3  4  5  6  7  8  9
Array: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                     ↑        ↑
                   start     end
       arr[5:8] gets elements 5,6,7
```
{% endstep %}
{% endstepper %}

## 2D Array Access: Matrix Magic! 🎯

{% stepper %}
{% step %}
### Creating a 2D Array
```python
arr2d = np.array([
    [1, 2, 3],  # Row 0
    [4, 5, 6],  # Row 1
    [7, 8, 9]   # Row 2
])
```
{% endstep %}

{% step %}
### Getting Elements
```python
# Get a row
print(arr2d[1])     # [4 5 6]

# Get single element
print(arr2d[1, 2])  # 6 (row 1, column 2)
print(arr2d[1][2])  # Same thing!
```
{% endstep %}

{% step %}
### Slicing 2D Arrays
```python
# First two rows
print(arr2d[:2])
# [[1 2 3]
#  [4 5 6]]

# First two rows, skip first column
print(arr2d[:2, 1:])
# [[2 3]
#  [5 6]]
```
{% endstep %}

{% step %}
### Visual Guide
```
       Columns (axis 1)
       0   1   2
Rows  ┌───────────┐
(axis │ 1  2  3   │ Row 0
 0)   │ 4  5  6   │ Row 1
      │ 7  8  9   │ Row 2
      └───────────┘
```
{% endstep %}
{% endstepper %}

![2d_array_indexing](./assets/ndarray_axis_index.png)

💡 **Pro Tips**:
- Use `:` to select everything in that dimension
- Remember: `[row, column]` order
- Slices create views (changes affect original)
- Think of 2D arrays like spreadsheets!
