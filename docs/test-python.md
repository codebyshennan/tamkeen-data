---
title: Python Code Examples
layout: default
---

# Python Code Examples

Here are some examples of Python code with syntax highlighting:

## Basic Python Example

```python
def greet(name):
    """
    A simple greeting function
    """
    return f"Hello, {name}!"

# Using the function
result = greet("World")
print(result)
```

## Data Analysis Example

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a sample DataFrame
df = pd.DataFrame({
    'A': np.random.randn(10),
    'B': np.random.randn(10)
})

# Calculate statistics
mean_a = df['A'].mean()
std_b = df['B'].std()

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(df['A'], df['B'])
plt.title('Scatter Plot of A vs B')
plt.xlabel('A')
plt.ylabel('B')
plt.show()
```

## Class Example

```python
class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.processed = False
    
    def process(self):
        """Process the data"""
        if not self.processed:
            self.data = self.data * 2
            self.processed = True
        return self.data
    
    @property
    def is_processed(self):
        return self.processed

# Using the class
processor = DataProcessor(np.array([1, 2, 3]))
result = processor.process()
print(f"Processed: {processor.is_processed}")
print(f"Result: {result}")
```
