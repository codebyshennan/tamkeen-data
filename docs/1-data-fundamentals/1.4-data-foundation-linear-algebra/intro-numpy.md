# Introduction to NumPy

## What is NumPy?

NumPy (Numerical Python) is like a supercharged calculator for Python! Imagine you need to do math on thousands or millions of numbers - NumPy makes it lightning fast and super easy. It's the foundation of scientific computing in Python and is used extensively in:

- Data Analysis: Processing large datasets efficiently
- Machine Learning: Building and training models
- Financial Analysis: Computing complex financial metrics
- Scientific Research: Processing experimental data
- Game Development: Handling physics calculations
- Image Processing: Manipulating pixels in images

{% stepper %}
{% step %}

### The Problem NumPy Solves

Let's see why regular Python lists aren't ideal for numerical computations:

```python
# Without NumPy (slow!) - Let's time it
import time

# Create a big list
numbers = list(range(1000000))
start_time = time.time()

# Double each number (need explicit loop)
doubled = [x * 2 for x in numbers]  # Need a loop 
python_time = time.time() - start_time

# With NumPy (fast! )
import numpy as np
numbers_np = np.array(range(1000000))
start_time = time.time()

# Double each number (vectorized operation)
doubled_np = numbers_np * 2  # No loop needed! 
numpy_time = time.time() - start_time

print(f"Python time: {python_time:.4f} seconds")
print(f"NumPy time: {numpy_time:.4f} seconds")
print(f"NumPy is {python_time/numpy_time:.1f}x faster!")
```

Key advantages of NumPy:

1. Vectorization: Operates on entire arrays at once
2. Contiguous Memory: Data stored efficiently
3. Low-level Optimization: Written in C for speed
4. Rich Functionality: Many mathematical operations built-in
{% endstep %}

{% step %}

### Cool Things NumPy Can Do

1. Lightning-fast calculations
2. Handle multi-dimensional data
3. Complex math made simple
4. Efficient memory usage
5. Works with other data science tools
{% endstep %}
{% endstepper %}

## Why is NumPy So Fast?

{% stepper %}
{% step %}

### Memory Magic

```
Python List:
[1] -> [2] -> [3] -> [4]  # Scattered in memory

NumPy Array:
[1,2,3,4]  # All together in one place!
```

- Like having all your tools on one table
- Everything is organized and easy to find
{% endstep %}

{% step %}

### Vectorization Power

Instead of:

```python
# Slow way (loops)
for i in range(1000):
    result[i] = numbers[i] * 2
```

NumPy way:

```python
# Fast way (vectorized)
result = numbers * 2  # All at once! 
```

{% endstep %}
{% endstepper %}

![numpy_vs_list](./assets/numpy_vs_python_list.png)

## Getting Started with NumPy

{% stepper %}
{% step %}

### Installation

Choose your way:

```bash
# Using pip
pip install numpy

# Using conda
conda install numpy
```

{% endstep %}

{% step %}

### Your First NumPy Program

```python
# Import NumPy (everyone uses 'np')
import numpy as np

# Create an array
numbers = np.array([1, 2, 3, 4, 5])

# Do magic! 
doubled = numbers * 2
squared = numbers ** 2
```

{% endstep %}
{% endstepper %}

## Speed Comparison

Let's race Python lists against NumPy arrays!

{% stepper %}
{% step %}

### The Setup

```python
# Create big numbers
python_list = list(range(1_000_000))
numpy_array = np.arange(1_000_000)
```

{% endstep %}

{% step %}

### The Race

```python
# Python List (slow! )
%timeit [x * 2 for x in python_list]
# Output: 100 ms ± 10 ms per loop

# NumPy Array (zoom! )
%timeit numpy_array * 2
# Output: 1 ms ± 0.1 ms per loop
```

That's 100 times faster!
{% endstep %}
{% endstepper %}

 **Pro Tips**:

- Always use `np` as the alias when importing NumPy
- Use NumPy when working with large amounts of numerical data
- Think in terms of operations on entire arrays, not individual elements
