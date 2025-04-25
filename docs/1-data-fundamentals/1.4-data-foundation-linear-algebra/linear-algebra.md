# Linear Algebra: Matrix Magic

## Set Operations: Finding Unique Values

{% stepper %}
{% step %}

### Finding Unique Values

Like finding unique cards in a deck! Set operations are crucial for:

- Data cleaning (removing duplicates)
- Feature engineering (unique categories)
- Finding distinct values
- Comparing datasets

Real-world applications:

- Finding unique customer IDs
- Extracting unique product categories
- Identifying unique transaction types
- Finding unique locations in data

```python
import numpy as np

# Array with duplicates
names = np.array(["Bob", "Will", "Joe", "Bob", "Will", "Joe", "Joe"])

# Get unique values and their counts
unique_names, counts = np.unique(names, return_counts=True)
print("Unique names:", unique_names)
print("Counts:", counts)
# Shows how many times each name appears

# Numbers work too!
numbers = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
unique_nums, counts = np.unique(numbers, return_counts=True)
print("\nUnique numbers:", unique_nums)
print("Counts:", counts)

# Real-world example - Sales data
sales_data = np.array([
    ['Electronics', 100],
    ['Books', 50],
    ['Electronics', 75],
    ['Clothing', 60],
    ['Books', 45]
])
unique_categories = np.unique(sales_data[:, 0])  # Get unique product categories
print("\nUnique product categories:", unique_categories)
```

{% endstep %}

{% step %}

### Testing Membership

Like checking if someone's on the guest list!

```python
# Guest list scenario
guests = np.array([1, 2, 3, 4, 5])  # Authorized IDs
to_check = np.array([2, 3, 6])      # IDs to verify

# Check membership
is_authorized = np.in1d(to_check, guests)
print("Authorization check:", is_authorized)
print("Authorized IDs:", to_check[is_authorized])
print("Unauthorized IDs:", to_check[~is_authorized])

# Real-world example - Product inventory
inventory = np.array(['SKU001', 'SKU002', 'SKU003', 'SKU004'])
orders = np.array(['SKU002', 'SKU005', 'SKU001'])

# Check which orders can be fulfilled
can_fulfill = np.in1d(orders, inventory)
print("\nOrder fulfillment check:")
print("Can fulfill:", orders[can_fulfill])
print("Out of stock:", orders[~can_fulfill])
```

{% endstep %}
{% endstepper %}

## Matrix Multiplication: The Dance of Numbers

{% stepper %}
{% step %}

### What is Matrix Multiplication?

Think of it like a special dance between numbers:

```python
# First matrix (2 rows, 3 columns)
x = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Second matrix (3 rows, 2 columns)
y = np.array([
    [6, 23],
    [-1, 7],
    [8, 9]
])

# Let them dance!
result = x @ y  # or np.dot(x, y)
print(result)
```

{% endstep %}

{% step %}

### Visual Guide to Matrix Multiplication

```
Matrix 1:      Matrix 2:      Result:
┌─────────┐    ┌───────┐     ┌───────┐
│ 1 2 3   │    │ 6  23 │     │ 28 84 │
│ 4 5 6   │  × │-1   7 │  =  │ 73 210│
└─────────┘    │ 8   9 │     └───────┘
               └───────┘

First element (28) = 1×6 + 2×(-1) + 3×8
```

{% endstep %}

{% step %}

### Three Ways to Multiply

```python
# Method 1: Using @
result1 = x @ y

# Method 2: Using dot
result2 = np.dot(x, y)

# Method 3: Using matmul
result3 = np.matmul(x, y)

# All give the same result!
```

{% endstep %}
{% endstepper %}

## Linear Algebra Operations: The Toolbox

{% stepper %}
{% step %}

### Common Operations

```python
# Create a 2x2 matrix
a = np.array([[1, 2],
              [3, 4]])

# Find determinant
det = np.linalg.det(a)
print("Determinant:", det)

# Find inverse
inv = np.linalg.inv(a)
print("Inverse:\n", inv)

# Solve linear equations
b = np.array([5, 11])
x = np.linalg.solve(a, b)
print("Solution:", x)
```

{% endstep %}

{% step %}

### Visual Guide to Operations

```
Original Matrix:   Inverse Matrix:
┌─────┐           ┌──────────┐
│ 1 2 │    =>     │ -2   1  │
│ 3 4 │           │ 1.5 -0.5│
└─────┘           └──────────┘

Solving equations:
1x + 2y = 5
3x + 4y = 11
Solution: x = 1, y = 2
```

{% endstep %}
{% endstepper %}

## Matrix Properties: Getting to Know Your Data

{% stepper %}
{% step %}

### Finding Matrix Properties

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

# Shape
print("Shape:", matrix.shape)  # (2, 3)

# Rank
print("Rank:", np.linalg.matrix_rank(matrix))

# Trace (sum of diagonal elements)
square_matrix = np.array([[1, 2],
                         [3, 4]])
print("Trace:", np.trace(square_matrix))
```

{% endstep %}

{% step %}

### Eigenvalues and Eigenvectors

```python
# For square matrices only
square = np.array([[4, -2],
                   [1, 1]])

# Get eigenvalues and eigenvectors
eigenvals, eigenvecs = np.linalg.eig(square)
print("Eigenvalues:", eigenvals)
print("Eigenvectors:\n", eigenvecs)
```

{% endstep %}
{% endstepper %}

 **Pro Tips**:

- Use `@` for matrix multiplication - it's cleaner!
- Check matrix shapes before multiplying
- Remember: not all matrices have inverses
- Use `np.linalg` for advanced operations
- Think about what operation makes sense for your data
