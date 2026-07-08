# Understanding Pandas Series

**After this lesson:** You can create a **Series**, interpret its index and **dtype**, and perform basic math and summary stats, thinking of it as one labeled column.

## Overview

**Prerequisites:** [Introduction to Python](../1.2-intro-python/README.md) and [Introduction to NumPy](../1.4-data-foundation-linear-algebra/intro-numpy.md) (arrays) at a basic level.

**Why this lesson:** A **Series** is a single column with an **index** and a **dtype**. Mastering it now makes **DataFrame** operations (selection, alignment, missing data) much easier later.

### Video

<div class="video-embed">
<iframe width="560" height="315" src="https://www.youtube.com/embed/zmdjNSmRXF4" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

*Corey Schafer, Python pandas tutorial (part 2): DataFrame and Series basics*

## What is a Series?

A Pandas Series is like a column in a spreadsheet or a single list of data with labels. Think of it as a smart, one-dimensional array that knows the name of each item! It's perfect for:

{% include mermaid-diagram.html src="1-data-fundamentals/1.5-data-analysis-pandas/diagrams/series-1.mmd" %}

*An index label sits next to each value, this is what makes a Series smarter than a plain Python list.*

- Time series data (stock prices over time)
- Storing categorical data (product categories)
- Tracking measurements (temperatures, distances)
- Working with labeled data (student grades)

Real-world applications:

- Financial data analysis
- Daily temperature readings
- Survey responses
- Sales performance tracking

---

### Creating Your First Series

we will look at different ways to create a Series:

**Construct Series from list, array, scalar, and custom index**

- **Purpose:** Recognize the three inputs pandas accepts for a 1-D Series and see default `RangeIndex` vs named day labels.
- **Walkthrough:** `pd.Series(list)`, `pd.Series(ndarray)`, broadcast `pd.Series(5, index=[...])`, then `temperatures.mean()`/`max()`.

```python
import pandas as pd
import numpy as np

# From a list
numbers = pd.Series([5, 6, -3, 2])
print("Simple Series:")
print(numbers)
print("\nType:", numbers.dtype)
print("Size:", numbers.size)
print("Shape:", numbers.shape)

# From a NumPy array
array_data = np.array([1.1, 2.2, 3.3, 4.4])
float_series = pd.Series(array_data)
print("\nFrom NumPy array:")
print(float_series)

# From a scalar value
constant = pd.Series(5, index=['a', 'b', 'c'])
print("\nConstant value Series:")
print(constant)

# Real-world example - Daily temperatures
temperatures = pd.Series([20.5, 22.1, 23.4, 21.8, 20.9],
                        index=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])
print("\nWeekly Temperatures:")
print(temperatures)
print("\nAverage temperature:", temperatures.mean())
print("Highest temperature:", temperatures.max())
```

```
Simple Series:
0    5
1    6
2   -3
3    2
dtype: int64

Type: int64
Size: 4
Shape: (4,)

From NumPy array:
0    1.1
1    2.2
2    3.3
3    4.4
dtype: float64

Constant value Series:
a    5
b    5
c    5
dtype: int64

Weekly Temperatures:
Mon    20.5
Tue    22.1
Wed    23.4
Thu    21.8
Fri    20.9
dtype: float64

Average temperature: 21.74
Highest temperature: 23.4
```

Notice how Pandas automatically creates numbered labels (0, 1, 2, 3) for each value!

---

### Custom Labels

You can create your own labels (called an index) for each value:

**String index for student grades**

- **Purpose:** Use non-numeric index labels so you can refer to rows by name (`grades['Alice']` in the next section).
- **Walkthrough:** `index=[...]` aligns 1:1 with the value list.

```python
# Create a Series with custom labels
grades = pd.Series([90, 85, 95, 78],
                  index=['Alice', 'Bob', 'Charlie', 'David'])
print(grades)
```

```
Alice      90
Bob        85
Charlie    95
David      78
dtype: int64
```

Output:

```
Alice      90
Bob        85
Charlie    95
David      78
dtype: int64
```

Now you can access values using these friendly names!

---

### Working with Series

Here are some common ways to work with your Series:

**Label, fancy index, and boolean mask**

- **Purpose:** Read one label, several labels, and filter by a condition on the Series values.
- **Walkthrough:** `grades[['Alice', 'Bob']]` passes a list of labels; `grades >= 90` returns a boolean Series aligned to `grades`.

```python
# Access a single value using its label
print(f"Charlie's grade: {grades['Charlie']}")  # Output: 95

# Get multiple values
print(grades[['Alice', 'Bob']])

# Filter values
passing_grades = grades[grades >= 90]
print("\nStudents with A grades:")
print(passing_grades)
```

```
Charlie's grade: 95
Alice    90
Bob      85
dtype: int64

Students with A grades:
Alice      90
Charlie    95
dtype: int64
```

## Series from Dictionary

---

### Creating from Dictionary

A Series can be created from a dictionary, where:

- Dictionary keys become the index (labels)
- Dictionary values become the Series values

**Series from mapping**

- **Purpose:** See how dict **keys → index** and **values → data**-common when you already have a lookup table in Python.
- **Walkthrough:** Order of rows follows insertion order (Python 3.7+ dicts).

```python
# Create a Series from a dictionary
population = pd.Series({
    'New York': 8.4,
    'London': 9.0,
    'Tokyo': 37.4,
    'Paris': 2.2
})
print(population)
```

```
New York     8.4
London       9.0
Tokyo       37.4
Paris        2.2
dtype: float64
```

Output:

```
New York     8.4
London       9.0
Tokyo       37.4
Paris        2.2
dtype: float64
```

---

### Converting Back to Dictionary

You can convert your Series back to a dictionary:

**Round-trip to plain Python dict**

- **Purpose:** Export a Series to a built-in dict for APIs or serialization that expect `{}`.
- **Walkthrough:** `population.to_dict()` maps index → value.

```python
# Convert Series to dictionary
pop_dict = population.to_dict()
print(pop_dict)
```

```
{'New York': 8.4, 'London': 9.0, 'Tokyo': 37.4, 'Paris': 2.2}
```

Output:

```python
{'New York': 8.4, 'London': 9.0, 'Tokyo': 37.4, 'Paris': 2.2}
```

```
{'New York': 8.4, 'London': 9.0, 'Tokyo': 37.4, 'Paris': 2.2}
```

## Working with Missing Data

---

### Understanding Missing Data

In the real world, data is often incomplete. Pandas uses `NaN` (Not a Number) to represent missing values:

**`None` becomes float NaN**

- **Purpose:** Observe how pandas stores missing entries and promotes dtypes when needed (`None` → `NaN`, often float column).
- **Walkthrough:** `Science: None` shows up as `NaN` in `print(scores)`.

```python
# Series with missing data
scores = pd.Series({'Math': 90, 'English': 85, 'Science': None, 'History': 88})
print(scores)
```

```
Math       90.0
English    85.0
Science     NaN
History    88.0
dtype: float64
```

Output:

```
Math       90.0
English    85.0
Science     NaN
History    88.0
dtype: float64
```

---

### Handling Missing Data

Pandas provides tools to work with missing data:

**Detect, drop, or fill NaNs**

- **Purpose:** Use `isna`, `dropna`, and `fillna` as the minimal toolkit before aggregating.
- **Walkthrough:** `scores.fillna(0)` is illustrative, choose a fill rule that matches your domain.

```python
# Check for missing values
print("Missing values?")
print(scores.isna())

# Drop missing values
print("\nScores without missing values:")
print(scores.dropna())

# Fill missing values
print("\nScores with filled values (0):")
print(scores.fillna(0))
```

```
Missing values?
Math       False
English    False
Science     True
History    False
dtype: bool

Scores without missing values:
Math       90.0
English    85.0
History    88.0
dtype: float64

Scores with filled values (0):
Math       90.0
English    85.0
Science     0.0
History    88.0
dtype: float64
```

## Series Operations

---

### Basic Math Operations

Series support mathematical operations, just like regular numbers:

**Broadcast scalar addition**

- **Purpose:** See element-wise `grades + 5` with **index preserved**-the same rule as NumPy broadcasting, but with labels.
- **Walkthrough:** `Alice`/`Bob`/`Charlie` keys stay aligned.

```python
# Original grades
grades = pd.Series({
    'Alice': 85,
    'Bob': 90,
    'Charlie': 78
})

# Add 5 points to everyone's grade
curved_grades = grades + 5
print("Grades after curve:")
print(curved_grades)
```

```
Grades after curve:
Alice      90
Bob        95
Charlie    83
dtype: int64
```

---

### Statistical Operations

Pandas provides many built-in statistical methods:

**Summary stats on labeled Series**

- **Purpose:** Use `mean`, `max`, `min`, and `describe()` for a quick numeric profile of one column.
- **Walkthrough:** `grades.describe()` returns count/mean/std/quartiles/min/max for numeric values.

```python
print(f"Average grade: {grades.mean()}")
print(f"Highest grade: {grades.max()}")
print(f"Lowest grade: {grades.min()}")
print(f"Grade summary:\n{grades.describe()}")
```

```
Average grade: 84.33333333333333
Highest grade: 90
Lowest grade: 78
Grade summary:
count     3.000000
mean     84.333333
std       6.027714
min      78.000000
25%      81.500000
50%      85.000000
75%      87.500000
max      90.000000
dtype: float64
```

## Best Practices and Tips

1. **Always Label Your Data**: Using meaningful index labels makes your data more readable and easier to work with.
2. **Check Data Types**: Use `dtype` to confirm your Series has the right data type.
3. **Handle Missing Values**: Always check for and handle missing values appropriately.
4. **Use Method Chaining**: You can combine operations like `grades.dropna().mean()`.

Remember: A Series is just the beginning. Once you're comfortable with Series, you'll be ready to tackle DataFrames, which are like multiple Series working together.

## Common pitfalls

- Forgetting that **Series** alignment is by **index**, operations pair labels, not just positions.
- Mixing up **loc** (label-based) and **iloc** (position-based) when you slice (covered in more detail in later lessons).
- Ignoring **NaN** values before calling **.mean()** or similar, check **.isna()** first when data is messy.

## Next steps

Continue to [Understanding DataFrames](./dataframe.md) to combine multiple columns and work with full tables.
