# Data Structures for Data Analysis

> **🎨 Essential Tool:** Open [Python Tutor](https://pythontutor.com) to visualize how data structures work in memory!

> **🤖 Start With AI:** Ask ChatGPT: "Explain Python data structures using everyday containers (boxes, folders, etc.)"

> **📓 Interactive Practice:** [Open in Google Colab](./notebooks/02-data-structures.ipynb)

## Introduction to Data Structures

### What Are Data Structures?

**Think of data structures as different types of containers for organizing information.**

Imagine you're organizing your belongings:
- A **backpack** (list) - You can add items, remove them, and they stay in order
- A **labeled filing cabinet** (dictionary) - You find things by their label, not by position
- A **sealed envelope** (tuple) - Once you put things in, you can't change them
- A **collection box** (set) - You just need to know if something is there or not, order doesn't matter

In Python, we use different "containers" (data structures) to store and organize our data efficiently. **Choosing the right container makes your code faster, clearer, and easier to work with.**

### Why Do We Need Different Data Structures?

**Short answer:** Different problems need different solutions!

**Example:**
- If you're keeping track of temperatures over time, you want them in order → Use a **list**
- If you're storing student names and their grades, you want to look up by name → Use a **dictionary**
- If you're tracking which products are in stock, you just need unique names → Use a **set**
- If you have coordinates (x, y) that shouldn't change, make them unchangeable → Use a **tuple**

> **💡 Key Insight:** Using the right data structure is like using the right tool. You can hammer a nail with a shoe, but a hammer works better! Similarly, you can solve problems with any data structure, but the right one makes life easier.

{% stepper %}
{% step %}

### The Four Main Data Structures You'll Use

Let's start with a simple overview before we dive deep into each one:

**1. Lists `[ ]` - Ordered Collections You Can Change**
```python
# A list is like a row of numbered boxes
shopping_list = ["milk", "eggs", "bread"]
# You can add more: shopping_list.append("cheese")
# You can change items: shopping_list[0] = "almond milk"
# Items stay in the order you put them
```

**2. Dictionaries `{ }` - Key-Value Pairs (Like a Real Dictionary!)**
```python
# A dictionary is like a real dictionary: word → definition
student = {
    "name": "Alice",      # key: "name", value: "Alice"
    "age": 20,           # key: "age", value: 20
    "grade": "A"         # key: "grade", value: "A"
}
# Look up by key: student["name"] gives you "Alice"
```

**3. Tuples `( )` - Ordered Collections You CANNOT Change**
```python
# A tuple is like a sealed envelope with data inside
coordinates = (10.5, 20.3)  # x and y position
# Once created, you can't change it: coordinates[0] = 15 → ERROR!
# Good for data that should never change (like GPS coordinates)
```

**4. Sets `{ }` - Unordered Collections of Unique Items**
```python
# A set is like a bag of unique items (no duplicates!)
unique_visitors = {"Alice", "Bob", "Charlie"}
# Adding "Alice" again does nothing (already in set)
# Order doesn't matter: {1, 2, 3} is the same as {3, 1, 2}
```

> **🎯 Remember:** 
> - **Lists** `[]` = Ordered, can change, allows duplicates
> - **Dictionaries** `{}` = Key-value pairs, fast lookup
> - **Tuples** `()` = Ordered, CANNOT change (immutable)
> - **Sets** `{}` = Unordered, unique values only

Now let's explore each one in detail!

{% endstep %}

{% step %}

### Real-World Data Science Example

Before we dive into details, let's see how these structures work together in a real scenario:

**Scenario: You're analyzing customer data for an online store**

```python
# DICTIONARY: Store complete customer information
customer = {
    "id": "C001",
    "name": "Alice Johnson",
    "email": "alice@email.com",
    "is_premium": True
}

# LIST: Track purchase history (order matters!)
purchase_history = [
    {"date": "2024-01-15", "amount": 49.99},
    {"date": "2024-01-20", "amount": 79.99},
    {"date": "2024-02-01", "amount": 29.99}
]

# TUPLE: Store fixed record (date, product, price)
sale_record = ("2024-01-15", "Laptop", 999.99)
# This should NEVER change (it's a historical record!)

# SET: Track unique product categories browsed
browsed_categories = {"Electronics", "Books", "Clothing", "Electronics"}
# Notice: Second "Electronics" is automatically ignored
# Result: {"Electronics", "Books", "Clothing"}
```

**Why each structure?**
- **Dictionary** for customer: We look up info by field name ("email", "name")
- **List** for purchases: Order matters (chronological history)
- **Tuple** for record: Historical data should never be modified
- **Set** for categories: We only care about unique values visited

This is how data scientists organize real data! Now let's learn each structure in depth.

{% endstep %}

{% step %}

### Data Structures in Data Science

Each data structure serves specific purposes in data analysis:

```python
import numpy as np
import pandas as pd

# Lists: Time series data
stock_prices = [100.23, 101.45, 99.78, 102.34]

# Tuples: Fixed structure records
data_point = ('2023-01-01', 'AAPL', 173.57, 1000000)

# Sets: Unique categories
unique_symbols = {'AAPL', 'GOOGL', 'MSFT', 'AMZN'}

# Dictionaries: Feature mappings
feature_info = {
    'price': {'type': 'numeric', 'missing': 0.02},
    'volume': {'type': 'numeric', 'missing': 0.00},
    'sector': {'type': 'categorical', 'unique_values': 11}
}

# NumPy Arrays: Efficient numerical computations
prices_array = np.array(stock_prices)

# Pandas Series: Labeled data
prices_series = pd.Series(stock_prices, 
                         index=pd.date_range('2023-01-01', periods=4))
```

Each structure optimized for different operations:

- Lists for flexible data collection
- Tuples for immutable records
- Sets for unique value operations
- Dictionaries for key-based lookups
- NumPy arrays for numerical computations
- Pandas for labeled data analysis

> **🔬 Visualize Data Structures:**
> Paste this into Python Tutor to see how different structures are stored:
> ```python
> # Watch how each structure is created and stored
> my_list = [1, 2, 3]
> my_tuple = (1, 2, 3)
> my_set = {1, 2, 3}
> my_dict = {'a': 1, 'b': 2, 'c': 3}
> 
> # See what happens with operations
> my_list.append(4)
> # my_tuple.append(4)  # Uncomment to see error!
> my_set.add(4)
> my_dict['d'] = 4
> ```

> **🤖 AI Challenge:**
> Ask: "Create a table comparing lists, tuples, sets, and dictionaries with their pros/cons"

{% endstep %}

{% step %}

### Performance Considerations

Choose structures based on operation needs:

```python
import time
import numpy as np

# Comparing list vs. numpy array operations
def compare_performance(size=1000000):
    # Create data
    list_data = list(range(size))
    array_data = np.array(list_data)
    
    # List operations
    start = time.time()
    list_result = [x * 2 for x in list_data]
    list_time = time.time() - start
    
    # NumPy operations
    start = time.time()
    array_result = array_data * 2
    array_time = time.time() - start
    
    print(f"List time: {list_time:.4f} seconds")
    print(f"NumPy time: {array_time:.4f} seconds")
    print(f"NumPy is {list_time/array_time:.1f}x faster")

# Memory usage comparison
def compare_memory():
    import sys
    
    # Create equivalent data structures
    data = list(range(1000))
    list_mem = sys.getsizeof(data)
    array_mem = np.array(data).nbytes
    
    print(f"List memory: {list_mem} bytes")
    print(f"NumPy memory: {array_mem} bytes")
```

{% endstep %}
{% endstepper %}

## Lists in Data Analysis - The Most Versatile Structure

### What is a List?

**A list is like a numbered row of containers where you can store anything, in any order, and change it whenever you want.**

Think of it like:
- A playlist of songs (you can add, remove, reorder)
- A to-do list (you can check items off, add new tasks)
- A shopping cart (items stay in order, you can modify it)

**Key characteristics:**
1. **Ordered** - Items stay in the order you put them
2. **Mutable** - You can change, add, or remove items
3. **Allows duplicates** - You can have the same value multiple times
4. **Indexed** - Each item has a position number (starting from 0)

### Creating Your First List

```python
# Empty list - like an empty container
empty_list = []

# List of numbers - great for numerical data
temperatures = [72, 75, 68, 70, 73]
#               [0] [1] [2] [3] [4]  ← These are index positions

# List of strings - good for names, labels
students = ["Alice", "Bob", "Charlie"]

# Mixed types - Python allows this (but use carefully!)
mixed = [42, "hello", True, 3.14]

# List of lists (2D data) - like a spreadsheet
grades = [
    ["Alice", 95, 87, 92],    # Row 0
    ["Bob", 88, 91, 85],       # Row 1
    ["Charlie", 92, 95, 90]    # Row 2
]
```

> **📍 Important:** Python uses **0-based indexing**. The first item is at position 0, not 1!

### Accessing List Items - Finding What You Need

```python
fruits = ["apple", "banana", "cherry", "date", "elderberry"]

# Getting items by position (index)
first_fruit = fruits[0]      # "apple" (first item)
second_fruit = fruits[1]      # "banana"
last_fruit = fruits[-1]       # "elderberry" (last item)
second_to_last = fruits[-2]   # "date"

print(f"First: {first_fruit}")
print(f"Last: {last_fruit}")

# Why does -1 mean last?
# Positive numbers count from the start: 0, 1, 2, 3, 4
# Negative numbers count from the end: -5, -4, -3, -2, -1
```

**Think of it like this:**
- `fruits[0]` = "Give me the 1st item" (computers count from 0!)
- `fruits[-1]` = "Give me the last item" (no matter how long the list is)
- `fruits[2]` = "Give me the 3rd item"

### Slicing Lists - Getting Multiple Items at Once

**Slicing is like cutting a portion of your list.**

```python
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#         [0][1][2][3][4][5][6][7][8][9]  ← index positions

# Basic slicing: list[start:stop]
# Note: 'stop' is NOT included (it's "up to but not including")

first_three = numbers[0:3]    # [0, 1, 2] - items 0, 1, 2 (not 3!)
middle_items = numbers[3:7]   # [3, 4, 5, 6] - items 3, 4, 5, 6 (not 7!)

# Shortcuts
from_start = numbers[:4]      # [0, 1, 2, 3] - from beginning to index 4
to_end = numbers[5:]          # [5, 6, 7, 8, 9] - from index 5 to end
everything = numbers[:]       # [0, 1, 2, ... 9] - copy of entire list

# Every nth item: list[start:stop:step]
every_second = numbers[::2]   # [0, 2, 4, 6, 8] - every 2nd item
every_third = numbers[::3]    # [0, 3, 6, 9] - every 3rd item
reversed_list = numbers[::-1] # [9, 8, 7, ... 0] - reverse order!
```

> **💡 Memory Trick for Slicing:**
> - `[start:stop]` = "Start here, stop before here"
> - Think of index numbers as **positions between items**, not the items themselves:
>   ```
>   Items:    [ 'a' | 'b' | 'c' | 'd' ]
>   Indices:  0     1     2     3     4
>   ```
> - `[1:3]` means "from position 1 to position 3" = 'b' and 'c'

### Modifying Lists - Making Changes

```python
shopping = ["milk", "bread", "eggs", "butter"]

# 1. CHANGE an existing item
shopping[0] = "almond milk"  # Replace "milk" with "almond milk"
print(shopping)  # ["almond milk", "bread", "eggs", "butter"]

# 2. ADD items to the end
shopping.append("cheese")    # Add one item to end
print(shopping)  # [..., "butter", "cheese"]

# 3. INSERT at a specific position
shopping.insert(1, "bananas")  # Insert "bananas" at position 1
print(shopping)  # ["almond milk", "bananas", "bread", ...]
# Everything after position 1 shifts to the right!

# 4. REMOVE items
shopping.remove("bread")     # Remove first occurrence of "bread"
last_item = shopping.pop()   # Remove and return last item
item_at_2 = shopping.pop(2)  # Remove and return item at index 2

# 5. EXTEND - add multiple items at once
more_items = ["yogurt", "juice"]
shopping.extend(more_items)  # Add all items from more_items
# Same as: shopping = shopping + more_items
```

> **🔍 What's the difference?**
> - `append(item)` - Adds ONE item (even if it's a list!)
> - `extend(list)` - Adds ALL items from the list
> ```python
> list1 = [1, 2, 3]
> list1.append([4, 5])    # Result: [1, 2, 3, [4, 5]]  ← list inside list!
> 
> list2 = [1, 2, 3]
> list2.extend([4, 5])    # Result: [1, 2, 3, 4, 5]  ← items added individually
> ```

### Checking if Items Exist

```python
fruits = ["apple", "banana", "cherry"]

# Check if something is in the list
if "banana" in fruits:
    print("We have bananas!")  # This will print

if "mango" not in fruits:
    print("No mangoes available")  # This will print

# Get the position (index) of an item
position = fruits.index("cherry")  # Returns 2
print(f"Cherry is at position {position}")

# Count how many times an item appears
numbers = [1, 2, 3, 2, 4, 2, 5]
count = numbers.count(2)  # Returns 3 (number 2 appears 3 times)
```

### Common List Methods - Your Toolkit

```python
data = [5, 2, 8, 1, 9, 3]

# Sort the list (changes the original!)
data.sort()              # [1, 2, 3, 5, 8, 9] - ascending order
data.sort(reverse=True)  # [9, 8, 5, 3, 2, 1] - descending order

# Sort without changing original (use sorted())
original = [5, 2, 8, 1, 9]
sorted_version = sorted(original)  # sorted_version = [1, 2, 5, 8, 9]
print(original)  # Still [5, 2, 8, 1, 9] - unchanged!

# Reverse the list
data.reverse()  # Reverses in place

# Get length
length = len(data)  # How many items?

# Clear all items
data.clear()  # Now data = []
```

{% stepper %}
{% step %}

### Advanced List Operations for Data Analysis

Now that you understand the basics, let's see how data scientists use lists:

```python
# Time series manipulation
prices = [100.23, 101.45, 99.78, 102.34, 101.89]

# Calculate returns
returns = [
    ((prices[i] - prices[i-1]) / prices[i-1]) * 100
    for i in range(1, len(prices))
]

# Moving average
def moving_average(data, window=3):
    return [
        sum(data[i:i+window]) / window
        for i in range(len(data) - window + 1)
    ]

# Data cleaning
def clean_data(data):
    """Remove outliers using IQR method"""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    return [x for x in data if lower_bound <= x <= upper_bound]
```

 **Performance Tip**: For numerical computations, prefer NumPy arrays over lists!

> **🎯 Try This Experiment:**
> Run this in Google Colab to see the speed difference:
> ```python
> import numpy as np
> import time
> 
> # List approach
> start = time.time()
> result_list = [x * 2 for x in range(1000000)]
> print(f"List time: {time.time() - start:.4f}s")
> 
> # NumPy approach
> start = time.time()
> result_array = np.arange(1000000) * 2
> print(f"NumPy time: {time.time() - start:.4f}s")
> ```

> **🤖 Learn More:**
> Ask: "Why is NumPy faster than Python lists for numerical operations?"

{% endstep %}

{% step %}

### List Comprehensions - Python's Power Feature

**What is a list comprehension?**
It's a **shortcut way to create new lists** based on existing lists. Think of it as a "recipe" written in one line instead of multiple lines.

**Why use them?**
- Faster to write (once you learn them!)
- Easier to read (for experienced Python programmers)
- Often faster to execute
- Very common in data science code

### From Loop to Comprehension - Step by Step

Let's see how to transform a regular loop into a list comprehension:

**Example 1: Squaring Numbers**

```python
# Traditional way - using a loop
numbers = [1, 2, 3, 4, 5]
squared = []  # Start with empty list

for num in numbers:      # For each number
    result = num ** 2    # Square it
    squared.append(result)  # Add to list

print(squared)  # [1, 4, 9, 16, 25]

# ✨ List comprehension way - ONE LINE!
squared = [num ** 2 for num in numbers]
print(squared)  # [1, 4, 9, 16, 25] - same result!
```

**How to read it:**
```python
squared = [num ** 2 for num in numbers]
#          ↑        ↑
#          |        └─ "for each num in numbers"
#          └────────── "create num squared"
```

Think of it like English: "Make a list of **num squared** for each **num in numbers**"

### Breaking Down the Pattern

**The template:**
```python
new_list = [expression for item in old_list]
           ↑           ↑         ↑
           |           |         └─ Source: where items come from
           |           └─────────── Loop: how to get each item
           └─────────────────────── Transform: what to do with item
```

**More examples to understand the pattern:**

```python
# Example 1: Convert to uppercase
names = ["alice", "bob", "charlie"]
uppercase = [name.upper() for name in names]
# Result: ["ALICE", "BOB", "CHARLIE"]
# Reading: "make name.upper() for each name in names"

# Example 2: Get lengths
words = ["hi", "hello", "hey"]
lengths = [len(word) for word in words]
# Result: [2, 5, 3]
# Reading: "make len(word) for each word in words"

# Example 3: Math operations
prices = [10, 20, 30]
with_tax = [price * 1.08 for price in prices]
# Result: [10.8, 21.6, 32.4]
# Reading: "make price * 1.08 for each price in prices"
```

### Adding Conditions (Filtering)

You can add `if` conditions to **filter** which items to include:

**Template with condition:**
```python
new_list = [expression for item in old_list if condition]
           ↑           ↑                     ↑
           |           |                     └─ Filter: only if this is True
           |           └───────────────────────Loop through items
           └───────────────────────────────── Transform each item
```

**Examples:**

```python
# Example 1: Only even numbers
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = [num for num in numbers if num % 2 == 0]
# Result: [2, 4, 6, 8, 10]
# Reading: "make num for each num in numbers IF num is even"

# Example 2: Only positive numbers
data = [-5, 10, -3, 20, -1, 30]
positives = [num for num in data if num > 0]
# Result: [10, 20, 30]
# Reading: "keep num for each num in data IF num is greater than 0"

# Example 3: Only long names
names = ["Al", "Alice", "Bob", "Charlie"]
long_names = [name for name in names if len(name) > 3]
# Result: ["Alice", "Charlie"]
# Reading: "keep name for each name in names IF name length > 3"
```

### Transformation + Filtering Together

You can both **transform** AND **filter** in the same comprehension:

```python
# Square only the even numbers
numbers = [1, 2, 3, 4, 5, 6, 7, 8]
squared_evens = [num ** 2 for num in numbers if num % 2 == 0]
# Result: [4, 16, 36, 64]
# Step 1: Filter to evens [2, 4, 6, 8]
# Step 2: Square each one [4, 16, 36, 64]

# Uppercase only short names
names = ["alice", "bob", "charlie", "dan"]
short_upper = [name.upper() for name in names if len(name) <= 3]
# Result: ["BOB", "DAN"]
# Step 1: Filter names with 3 or fewer letters ["bob", "dan"]
# Step 2: Convert to uppercase ["BOB", "DAN"]
```

> **💡 How to think about it:**
> 1. Start with: "for item in list"
> 2. Add filter (if needed): "if condition"
> 3. Add transformation: "expression"
> 4. Wrap in brackets: `[expression for item in list if condition]`

### Real Data Science Examples

```python
# Example 1: Clean and convert temperature data
temp_strings = ["23.5", "25.0", "22.8", "invalid", "24.1"]

# Traditional way (multiple steps)
clean_temps = []
for temp in temp_strings:
    try:
        clean_temps.append(float(temp))
    except ValueError:
        pass  # Skip invalid values

# Better: List comprehension with helper function
def safe_float(s):
    try:
        return float(s)
    except ValueError:
        return None

temps = [safe_float(t) for t in temp_strings]
temps = [t for t in temps if t is not None]  # Remove None values
# Result: [23.5, 25.0, 22.8, 24.1]

# Example 2: Extract specific fields from dictionaries
students = [
    {"name": "Alice", "grade": 95},
    {"name": "Bob", "grade": 87},
    {"name": "Charlie", "grade": 92}
]

# Get just the names
names = [student["name"] for student in students]
# Result: ["Alice", "Bob", "Charlie"]

# Get names of students with grade > 90
top_students = [s["name"] for s in students if s["grade"] > 90]
# Result: ["Alice", "Charlie"]
```

### Common Mistakes to Avoid

```python
# ❌ WRONG: Forgetting brackets
numbers = [1, 2, 3]
squared = num ** 2 for num in numbers  # ERROR! Not a list!

# ✅ CORRECT: Include brackets
squared = [num ** 2 for num in numbers]

# ❌ WRONG: Condition in wrong place
evens = [num for num in numbers if num % 2 == 0 ** 2]  # Confusing!

# ✅ CORRECT: Transform after filtering
evens_squared = [num ** 2 for num in numbers if num % 2 == 0]

# ❌ WRONG: Too complex (hard to read!)
result = [x ** 2 + y ** 2 for x in range(10) for y in range(10) if x > y and x % 2 == 0]

# ✅ BETTER: Use traditional loop when it gets complex
result = []
for x in range(10):
    for y in range(10):
        if x > y and x % 2 == 0:
            result.append(x ** 2 + y ** 2)
```

> **🎯 Rule of Thumb:**
> - If your list comprehension fits comfortably on one line → Use it!
> - If it's getting complex and hard to read → Use a traditional loop
> - Readability matters more than showing off!

### Practice: Convert Loops to Comprehensions

Try converting these loops to comprehensions (answers at bottom):

```python
# Exercise 1: Traditional loop
fruits = ["apple", "banana", "cherry"]
lengths = []
for fruit in fruits:
    lengths.append(len(fruit))

# Your comprehension:
# lengths = [...]

# Exercise 2: Traditional loop with condition
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
odd_squares = []
for num in numbers:
    if num % 2 == 1:
        odd_squares.append(num ** 2)

# Your comprehension:
# odd_squares = [...]
```

<details>
<summary>💡 Answers</summary>

```python
# Exercise 1
lengths = [len(fruit) for fruit in fruits]

# Exercise 2
odd_squares = [num ** 2 for num in numbers if num % 2 == 1]
```
</details>

{% endstep %}

{% step %}

### List Comprehensions in Data Science

Now let's see advanced data science applications:

```python
import pandas as pd
import numpy as np

# Feature engineering
dates = ['2023-01-01', '2023-01-02', '2023-01-03']
values = [100, 101, 99]

# Create time features
features = [{
    'date': pd.to_datetime(date),
    'value': value,
    'year': pd.to_datetime(date).year,
    'month': pd.to_datetime(date).month,
    'day': pd.to_datetime(date).day,
    'day_of_week': pd.to_datetime(date).dayofweek
} for date, value in zip(dates, values)]

# Data normalization
def normalize_features(data):
    """Min-max normalization"""
    min_val = min(data)
    max_val = max(data)
    return [
        (x - min_val) / (max_val - min_val)
        if max_val > min_val else 0
        for x in data
    ]
```

{% endstep %}
{% endstepper %}

## Tuples in Data Analysis

{% stepper %}
{% step %}

### Efficient Data Records

Using tuples for fixed-structure data:

```python
# Dataset records
records = [
    ('2023-01-01', 'AAPL', 173.57, 1000000),
    ('2023-01-01', 'GOOGL', 2951.88, 500000),
    ('2023-01-01', 'MSFT', 339.32, 750000)
]

# Efficient unpacking
for date, symbol, price, volume in records:
    # Process each field
    pass

# Named tuples for better readability
from collections import namedtuple

StockRecord = namedtuple('StockRecord', 
                        ['date', 'symbol', 'price', 'volume'])

records = [
    StockRecord('2023-01-01', 'AAPL', 173.57, 1000000),
    StockRecord('2023-01-01', 'GOOGL', 2951.88, 500000)
]

# Access by name
print(records[0].price)  # 173.57
```

{% endstep %}

{% step %}

### Tuple Performance Advantages

Memory and speed benefits:

```python
import sys
from timeit import timeit

# Memory comparison
tuple_data = tuple(range(1000))
list_data = list(range(1000))

print(f"Tuple size: {sys.getsizeof(tuple_data)} bytes")
print(f"List size: {sys.getsizeof(list_data)} bytes")

# Performance comparison
def compare_access():
    # Setup
    setup = """
    tuple_data = tuple(range(1000))
    list_data = list(range(1000))
    """
    
    # Test tuple access
    tuple_time = timeit(
        'x = tuple_data[500]',
        setup=setup,
        number=1000000
    )
    
    # Test list access
    list_time = timeit(
        'x = list_data[500]',
        setup=setup,
        number=1000000
    )
    
    print(f"Tuple access time: {tuple_time:.6f} seconds")
    print(f"List access time: {list_time:.6f} seconds")
```

{% endstep %}
{% endstepper %}

## Sets in Data Analysis

{% stepper %}
{% step %}

### Advanced Set Operations

Efficient unique value operations:

```python
# Feature selection
numerical_features = {'price', 'volume', 'returns'}
categorical_features = {'sector', 'industry', 'exchange'}

# Find features present in both types
common_features = numerical_features & categorical_features

# Find unique features to each type
numerical_only = numerical_features - categorical_features
categorical_only = categorical_features - numerical_features

# Efficient unique value counting
def get_unique_counts(df):
    """Get unique value counts for each column"""
    return {
        col: len(set(df[col].dropna()))
        for col in df.columns
    }

# Duplicate detection
def find_duplicates(data):
    """Find duplicate values in a sequence"""
    seen = set()
    duplicates = set()
    
    for item in data:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    
    return duplicates
```

{% endstep %}

{% step %}

### Set Operations for Data Cleaning

Common data cleaning patterns:

```python
class DataCleaner:
    def __init__(self, df):
        self.df = df
        self.categorical_cols = set()
        self.numeric_cols = set()
        self._identify_column_types()
    
    def _identify_column_types(self):
        """Identify column types"""
        for col in self.df.columns:
            if np.issubdtype(self.df[col].dtype, np.number):
                self.numeric_cols.add(col)
            else:
                self.categorical_cols.add(col)
    
    def standardize_categories(self, columns=None):
        """Standardize categorical values"""
        columns = columns or self.categorical_cols
        
        for col in columns:
            # Get unique values
            unique_values = set(self.df[col].dropna())
            
            # Create mapping for similar values
            mapping = {}
            for value in unique_values:
                key = str(value).lower().strip()
                if key not in mapping:
                    mapping[key] = value
            
            # Apply standardization
            self.df[col] = self.df[col].apply(
                lambda x: mapping.get(
                    str(x).lower().strip(), x
                ) if pd.notna(x) else x
            )
```

{% endstep %}
{% endstepper %}

## Dictionaries in Data Analysis - The Lookup Master

### What is a Dictionary?

**A dictionary is like a real dictionary or phonebook - you look up a KEY to get its VALUE.**

Think of it like:
- **A real dictionary**: Look up a WORD (key) to find its DEFINITION (value)
- **A phonebook**: Look up a NAME (key) to find a PHONE NUMBER (value)
- **A locker room**: Each LOCKER NUMBER (key) opens a specific LOCKER (value)
- **A restaurant menu**: Find a DISH NAME (key) to see its PRICE (value)

**Key characteristics:**
1. **Key-Value pairs** - Every piece of data has a label (key) and content (value)
2. **Fast lookup** - Finding a value by its key is super fast (even in huge dictionaries!)
3. **Keys must be unique** - Each key can only appear once (but values can repeat)
4. **Unordered** - Items don't have a position/index like lists (they're organized by keys)

### Creating Your First Dictionary

```python
# Empty dictionary
empty_dict = {}
empty_dict2 = dict()  # Another way

# Simple dictionary: key → value
student = {
    "name": "Alice",      # key: "name", value: "Alice"
    "age": 20,           # key: "age", value: 20
    "grade": "A",        # key: "grade", value: "A"
    "is_active": True    # key: "is_active", value: True
}

# Think of it visually:
# ┌─────────────┬───────────┐
# │ Key         │ Value     │
# ├─────────────┼───────────┤
# │ "name"      │ "Alice"   │
# │ "age"       │ 20        │
# │ "grade"     │ "A"       │
# │ "is_active" │ True      │
# └─────────────┴───────────┘
```

### Why Use Dictionaries?

**Compare these two approaches:**

```python
# ❌ Using separate variables (messy!)
student_name = "Alice"
student_age = 20
student_grade = "A"
student_is_active = True

# If you have 100 students, you'd need 400 variables! 😱

# ✅ Using a dictionary (organized!)
student = {
    "name": "Alice",
    "age": 20,
    "grade": "A",
    "is_active": True
}

# All related data is in ONE place!
# Easy to pass around, easy to understand
```

### Accessing Dictionary Values

```python
student = {
    "name": "Alice",
    "age": 20,
    "major": "Computer Science",
    "gpa": 3.8
}

# Method 1: Using square brackets [key]
name = student["name"]      # "Alice"
age = student["age"]        # 20

print(f"{name} is {age} years old")
# Output: Alice is 20 years old

# Method 2: Using .get() method (safer!)
major = student.get("major")           # "Computer Science"
grade = student.get("grade", "N/A")    # "N/A" (key doesn't exist, returns default)

# ⚠️ What's the difference?
student["graduation_year"]      # ❌ ERROR! KeyError - key doesn't exist
student.get("graduation_year")  # ✅ Returns None (no error)
student.get("graduation_year", 2024)  # ✅ Returns 2024 (custom default)
```

> **💡 Pro Tip:** Use `.get()` when you're not sure if a key exists. Use `[key]` when you're certain it exists.

### Adding and Modifying Values

```python
person = {
    "name": "Bob",
    "age": 25
}

# 1. ADD new key-value pairs
person["city"] = "New York"      # Add city
person["occupation"] = "Engineer" # Add occupation

print(person)
# {"name": "Bob", "age": 25, "city": "New York", "occupation": "Engineer"}

# 2. MODIFY existing values
person["age"] = 26  # Change age from 25 to 26
person["city"] = "San Francisco"  # Change city

# 3. ADD multiple items at once
person.update({
    "email": "bob@email.com",
    "phone": "555-0123"
})

# 4. REMOVE items
removed_value = person.pop("phone")  # Remove "phone", returns its value
print(removed_value)  # "555-0123"

del person["email"]  # Another way to remove

# 5. Remove last item added (rarely used)
last_item = person.popitem()  # Returns tuple: (key, value)
```

### Dictionary Keys - Important Rules

**What can be a key?**

```python
# ✅ ALLOWED: Immutable types (strings, numbers, tuples)
valid_dict = {
    "name": "Alice",          # String key ✅
    42: "The Answer",          # Number key ✅
    (10, 20): "Coordinates",   # Tuple key ✅
    True: "Yes"                # Boolean key ✅
}

# ❌ NOT ALLOWED: Mutable types (lists, dictionaries, sets)
invalid_dict = {
    [1, 2]: "Bad"  # ❌ ERROR! Lists can't be keys
}

# Why? Keys must be unchangeable (immutable) so Python can find them quickly
```

**Keys must be unique:**
```python
# What happens with duplicate keys?
data = {
    "name": "Alice",
    "age": 20,
    "name": "Bob"  # ⚠️ Duplicate key!
}

print(data)
# {"name": "Bob", "age": 20}
# The second "name" OVERWRITES the first one!
```

### Checking What's in a Dictionary

```python
student = {
    "name": "Alice",
    "age": 20,
    "major": "CS"
}

# Check if a KEY exists
if "name" in student:
    print("Name is recorded")  # ✅ This prints

if "grade" in student:
    print("Grade is recorded")  # This won't print
else:
    print("No grade recorded")  # ✅ This prints

# Check if a key does NOT exist
if "email" not in student:
    print("No email on file")  # ✅ This prints

# Get all keys
keys = student.keys()
print(keys)  # dict_keys(['name', 'age', 'major'])
print(list(keys))  # ['name', 'age', 'major'] - convert to list

# Get all values
values = student.values()
print(list(values))  # ['Alice', 20, 'CS']

# Get all key-value pairs
items = student.items()
print(list(items))
# [('name', 'Alice'), ('age', 20), ('major', 'CS')]
# Each pair is a tuple!
```

### Looping Through Dictionaries

**Multiple ways to iterate:**

```python
student = {
    "name": "Alice",
    "age": 20,
    "major": "CS"
}

# Method 1: Loop through KEYS (default)
for key in student:
    print(f"Key: {key}")
# Output:
# Key: name
# Key: age
# Key: major

# Method 2: Loop through KEYS explicitly
for key in student.keys():
    print(f"{key} = {student[key]}")
# Output:
# name = Alice
# age = 20
# major = CS

# Method 3: Loop through VALUES
for value in student.values():
    print(f"Value: {value}")
# Output:
# Value: Alice
# Value: 20
# Value: CS

# Method 4: Loop through BOTH keys and values (most common!)
for key, value in student.items():
    print(f"{key}: {value}")
# Output:
# name: Alice
# age: 20
# major: CS
```

### Nested Dictionaries - Dictionaries Inside Dictionaries

**Think of this like a filing cabinet with folders inside folders:**

```python
# A company's employee data
employees = {
    "E001": {
        "name": "Alice",
        "department": "Engineering",
        "salary": 90000,
        "skills": ["Python", "SQL", "AWS"]
    },
    "E002": {
        "name": "Bob",
        "department": "Marketing",
        "salary": 75000,
        "skills": ["SEO", "Analytics"]
    }
}

# Accessing nested data - go level by level
alice = employees["E001"]           # Get Alice's entire record
alice_name = employees["E001"]["name"]    # Get Alice's name
alice_skills = employees["E001"]["skills"]  # Get Alice's skills list

print(f"{alice_name} knows {', '.join(alice_skills)}")
# Output: Alice knows Python, SQL, AWS

# Modifying nested data
employees["E001"]["salary"] = 95000  # Give Alice a raise!

# Adding to nested lists
employees["E001"]["skills"].append("Docker")  # Alice learned Docker!

# Safely accessing nested data
# ❌ This could crash if key doesn't exist:
salary = employees["E003"]["salary"]  # KeyError!

# ✅ Safe way:
salary = employees.get("E003", {}).get("salary", "Not Found")
print(salary)  # "Not Found"
```

### Real-World Example: Product Inventory

```python
# An online store's inventory system
inventory = {
    "PROD001": {
        "name": "Laptop",
        "price": 999.99,
        "stock": 15,
        "category": "Electronics",
        "specs": {
            "ram": "16GB",
            "storage": "512GB SSD",
            "screen": "15.6 inch"
        }
    },
    "PROD002": {
        "name": "Mouse",
        "price": 29.99,
        "stock": 50,
        "category": "Accessories",
        "specs": {
            "type": "Wireless",
            "dpi": "1600"
        }
    }
}

# Function to display product info
def show_product(product_id):
    if product_id in inventory:
        product = inventory[product_id]
        print(f"📦 {product['name']}")
        print(f"   Price: ${product['price']}")
        print(f"   In stock: {product['stock']} units")
        print(f"   Category: {product['category']}")
        print(f"   Specs: {product['specs']}")
    else:
        print(f"Product {product_id} not found")

# Function to check if product is available
def is_available(product_id, quantity):
    product = inventory.get(product_id)
    if product:
        return product["stock"] >= quantity
    return False

# Usage
show_product("PROD001")
print(f"Can buy 10 laptops? {is_available('PROD001', 10)}")  # True
print(f"Can buy 20 laptops? {is_available('PROD001', 20)}")  # False
```

### Dictionary Comprehensions - Yes, These Exist Too!

Just like list comprehensions, but for dictionaries:

```python
# Create dictionary from lists
keys = ["name", "age", "city"]
values = ["Alice", 25, "NYC"]

# Traditional way
person = {}
for i in range(len(keys)):
    person[keys[i]] = values[i]

# Dictionary comprehension way!
person = {keys[i]: values[i] for i in range(len(keys))}
# Better: use zip()
person = {k: v for k, v in zip(keys, values)}
print(person)  # {"name": "Alice", "age": 25, "city": "NYC"}

# More examples
# Square numbers as dictionary
squares = {num: num**2 for num in range(1, 6)}
print(squares)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# Filter dictionary by condition
prices = {"apple": 1.20, "banana": 0.50, "orange": 1.80, "grape": 2.50}
expensive = {fruit: price for fruit, price in prices.items() if price > 1.00}
print(expensive)  # {"apple": 1.20, "orange": 1.80, "grape": 2.50}

# Transform values
prices_with_tax = {fruit: price * 1.08 for fruit, price in prices.items()}
```

{% stepper %}
{% step %}

### Advanced Dictionary Patterns for Data Science

Now let's see professional data science applications:

```python
class DatasetMetadata:
    def __init__(self, df):
        self.df = df
        self.metadata = self._generate_metadata()
    
    def _generate_metadata(self):
        """Generate comprehensive dataset metadata"""
        metadata = {}
        
        for column in self.df.columns:
            metadata[column] = {
                'dtype': str(self.df[column].dtype),
                'missing_count': self.df[column].isna().sum(),
                'missing_percentage': (
                    self.df[column].isna().mean() * 100
                ),
                'unique_count': self.df[column].nunique(),
                'memory_usage': self.df[column].memory_usage(deep=True)
            }
            
            # Add type-specific metadata
            if np.issubdtype(self.df[column].dtype, np.number):
                metadata[column].update({
                    'mean': self.df[column].mean(),
                    'std': self.df[column].std(),
                    'min': self.df[column].min(),
                    'max': self.df[column].max()
                })
            else:
                metadata[column].update({
                    'most_common': self.df[column].mode().iloc[0],
                    'unique_values': list(
                        self.df[column].value_counts()
                        .head()
                        .to_dict()
                    )
                })
        
        return metadata
```

{% endstep %}

{% step %}

### Dictionary Comprehensions for Analysis

Efficient data transformations:

```python
# Feature statistics
def calculate_feature_stats(df):
    """Calculate statistics for each numeric feature"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    return {
        col: {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'skew': df[col].skew(),
            'kurtosis': df[col].kurtosis()
        }
        for col in numeric_cols
    }

# Correlation analysis
def analyze_correlations(df, threshold=0.7):
    """Find highly correlated feature pairs"""
    corr_matrix = df.corr()
    
    return {
        (col1, col2): corr_matrix.loc[col1, col2]
        for col1 in corr_matrix.columns
        for col2 in corr_matrix.columns
        if col1 < col2 and  # Avoid duplicates
        abs(corr_matrix.loc[col1, col2]) >= threshold
    }
```

{% endstep %}
{% endstepper %}

## 🎯 Practice Exercises for Data Analysis

> **💡 Learning Strategy:** Solve in Colab → Visualize in Python Tutor → Review with AI

### Exercise 1: List Manipulation
```python
# Create a list of stock prices
prices = [100, 102, 98, 105, 103, 107, 110]

# Tasks:
# 1. Calculate the daily price changes
# 2. Find the highest and lowest prices
# 3. Identify days where price increased
# 4. Calculate average price

# Start here:
# Your code...
```

> **🔍 Visualize:** Use Python Tutor to see how list comprehensions work
> **🤖 Help:** Ask AI: "Explain list comprehensions with simple examples"

### Exercise 2: Dictionary for Data Aggregation
```python
# Sales data by category
sales = [
    {'category': 'Electronics', 'amount': 1200},
    {'category': 'Clothing', 'amount': 800},
    {'category': 'Electronics', 'amount': 1500},
    {'category': 'Food', 'amount': 300},
    {'category': 'Clothing', 'amount': 600},
]

# Tasks:
# 1. Calculate total sales per category
# 2. Find the category with highest sales
# 3. Calculate average sale amount per category

# Your code here...
```

> **🎨 Visualize This:** See how dictionaries accumulate values in Python Tutor
> **🤖 Prompt:** "Show me 3 ways to aggregate data using Python dictionaries"

### Exercise 3: Set Operations for Data Analysis
```python
# Customer data from two campaigns
campaign_a = {'customer1', 'customer2', 'customer3', 'customer4'}
campaign_b = {'customer3', 'customer4', 'customer5', 'customer6'}

# Tasks:
# 1. Find customers in both campaigns (intersection)
# 2. Find all unique customers (union)
# 3. Find customers only in campaign A
# 4. Find customers only in campaign B

# Your code here...
```

> **🔬 Experiment:** Watch set operations in Python Tutor - it's beautiful!
> **🤖 Ask:** "Explain Venn diagrams and how they relate to Python sets"

### Exercise 4: Nested Data Structures
```python
# Student records
students = [
    {'name': 'Alice', 'grades': [85, 90, 88], 'major': 'CS'},
    {'name': 'Bob', 'grades': [78, 82, 80], 'major': 'Math'},
    {'name': 'Charlie', 'grades': [92, 95, 94], 'major': 'CS'},
]

# Tasks:
# 1. Calculate average grade for each student
# 2. Find CS majors with average > 85
# 3. Create a dictionary mapping name to average grade
# 4. Find the student with highest average

# Your code here...
```

> **🎯 Advanced:** Paste this into Python Tutor and step through nested loops
> **🤖 Challenge:** Ask AI: "Create a similar exercise with company sales data"

## 🚀 Bonus Challenges

### Challenge 1: Data Structure Olympics
Pick the BEST data structure for each scenario and explain why:
1. Storing unique visitor IDs
2. Mapping product codes to prices
3. Storing ordered transaction history
4. Caching computed results

> **🤖 Validate:** Ask AI to review your choices and reasoning

### Challenge 2: Build a Mini Database
Create a simple contact management system using dictionaries:
- Add contacts
- Search by name
- Update phone numbers
- List all contacts

> **📺 Need Help?** Check [Video Resources](./video-resources.md) - Section on Dictionaries

Remember:

- Choose appropriate data structures for your task
- Consider performance implications
- Handle edge cases
- Document your code
- **Visualize complex operations in Python Tutor**
- **Use AI to understand trade-offs between structures**

Happy analyzing!
