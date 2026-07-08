# Conditions and Iterations in Data Analysis

**After this lesson:** you can explain Conditions and Iterations in Data Analysis and try the examples in your own notebook.

> **Best for Visualization:** Loops and conditions are AMAZING in Python Tutor - watch the flow!

> **AI Starter:** "Explain if-else statements using real-world decision-making examples"

> **Practice:** Run the examples locally or in [Google Colab](https://colab.research.google.com). This submodule ships notebooks for [basic syntax](notebooks/01-basic-syntax.ipynb), [data structures](notebooks/02-data-structures.ipynb), and [functions](notebooks/03-functions.ipynb); paste loop and branch examples there if you want a notebook environment.

### Video

_Corey Schafer, Loops and iteration in Python_

**How this fits together:** `if` / `elif` / `else` choose what runs once; `for` and `while` repeat work. Data pipelines use both: **validate** a row with branches, **scan** a table with loops, or prefer vectorized NumPy/pandas later. Master the ideas here so you can read any script that filters, iterates, or retries.

_`if/elif/else` picks one branch per value; a `for` loop runs the same block for every item in a collection._

## Making Decisions with Conditions

***

### Understanding If Statements in Data Analysis

Conditions are important for data filtering and validation:

```python
import pandas as pd
import numpy as np

# Data validation example
def validate_age(age):
   if age < 0:
       return np.nan  # Invalid age
   elif age > 120:
       return np.nan  # Likely invalid age
   else:
       return age

# Handling missing values
def process_value(value):
   if pd.isna(value):
       return 0  # Replace missing with default
   elif np.isinf(value):
       return np.nan  # Handle infinity
   else:
       return value
```

**Remember**: Always validate your data before analysis!

> **Watch Control Flow:** Paste this into Python Tutor - watch which branch executes!
>
> ```python
> age = 25
>
> if age < 18:
>   category = "Minor"
>   discount = 0.1
> elif age < 65:
>   category = "Adult"
>   discount = 0
> else:
>   category = "Senior"
>   discount = 0.2
>
> print(f"{category}: {discount * 100}% discount")
> ```

```
Adult: 0% discount
```

> **Experiment:** Ask AI: "Create 5 real-world scenarios that use if-elif-else"

***

### If-Else in Data Processing

Common data processing scenarios:

```python
import pandas as pd

# Data quality check
def check_data_quality(df):
   if df.isnull().sum().any():
       print("Warning: Dataset contains missing values")
       missing_stats = df.isnull().sum()
       print(f"Missing value counts:\n{missing_stats}")
   else:
       print("Data quality check passed: No missing values")

# Outlier detection
def flag_outlier(value, mean, std):
   if abs(value - mean) > 3 * std:
       return 'outlier'
   else:
       return 'normal'
```

Real-world example:

```python
# Sales data analysis
def analyze_sales_performance(sales_value, target):
   if sales_value >= target * 1.2:
       return 'Exceptional'
   elif sales_value >= target:
       return 'Met Target'
   elif sales_value >= target * 0.8:
       return 'Near Target'
   else:
       return 'Below Target'
```

***

### Multiple Conditions in Data Analysis

Complex data processing decisions:

Imports

Both pandas and NumPy are imported at the top since the functions below rely on pandas null checks and NumPy NaN.

Customer Segmentation

Combines `and` / `or` and nested `if` to segment customers into four tiers based on spend, frequency, and tenure.

Type-Aware Transform

Guards against null and infinity first, then branches on `data_type` to coerce numeric strings or normalise categorical strings.

***

### Nested Conditions in Feature Engineering

Complex feature creation:

Nested Age Classifier

The inner function handles a null guard first, then branches on gender, then on age brackets, three levels of nesting for six distinct labels.

Row-Wise Apply

Uses `df.apply(..., axis=1)` to call the inner function once per row, passing both the `age` and `gender` columns as a unit.

## Data Filtering and Comparison

***

### Comparison Operations in Pandas

Efficient data filtering:

```python
import pandas as pd
import numpy as np

# Load sample data
df = pd.DataFrame({
   'value': [10, 20, 30, 40, 50],
   'category': ['A', 'B', 'A', 'B', 'C']
})

# Single condition
high_values = df[df['value'] > 30]

# Multiple conditions
filtered_data = df[
   (df['value'] > 20) &
   (df['category'] == 'A')
]

# Complex filtering
def filter_outliers(df, columns, n_std=3):
   """Filter outliers based on standard deviation"""
   for col in columns:
       mean = df[col].mean()
       std = df[col].std()
       df = df[
           (df[col] >= mean - n_std * std) &
           (df[col] <= mean + n_std * std)
       ]
   return df
```

**Performance Tip**: Use vectorized operations instead of loops for filtering!

> **Speed Comparison:** Run this in Google Colab to see the difference:
>
> ```python
> import pandas as pd
> import numpy as np
> import time
>
> # Create large dataset
> df = pd.DataFrame({'value': np.random.randint(0, 100, 1000000)})
>
> # Slow: Loop approach
> start = time.time()
> result = []
> for val in df['value']:
>   if val > 50:
>     result.append(val)
> loop_time = time.time() - start
>
> # Fast: Vectorized approach
> start = time.time()
> result = df[df['value'] > 50]
> vector_time = time.time() - start
>
> print(f"Loop: {loop_time:.4f}s")
> print(f"Vectorized: {vector_time:.4f}s")
> print(f"Speedup: {loop_time/vector_time:.1f}x faster!")
> ```

```
Loop: 0.0535s
Vectorized: 0.0029s
Speedup: 18.6x faster!
```

> **Learn Why:** Ask: "Why are vectorized operations faster than loops in pandas?"

***

### Logical Operations in Data Analysis

Combining multiple conditions:

Import

Only pandas is imported here; NumPy is used via `np` from the surrounding module scope.

Validity Check

Evaluates four quality conditions into a dict, then loops over any that are True to print a labelled report, returns False if any issue is found.

Outlier Detection

Computes Z-scores per numeric column and short-circuits with `break` on the first column that exceeds the threshold, avoiding unnecessary work.

## Efficient Data Iteration

***

### Vectorized Operations vs. Loops

Understanding performance implications:

Loop Approach

Iterates row-by-row with `iterrows()`, branching on sign to compute `log` or append NaN, correct but slow for large DataFrames.

Vectorized Equivalent

`np.where` applies the same condition across the entire column at once, no Python loop, so typically 10-100x faster.

Multi-Label Select

Computes a Z-score column then uses `np.select` with three boolean masks to assign Low / Normal / High labels in a single vectorised pass.

***

### Efficient Iteration When Necessary

Some cases require iteration:

Imports

Imports `tqdm` alongside pandas to wrap the chunk loop in a progress bar so long-running jobs show their progress.

Chunk Iteration

Steps through the DataFrame in `chunk_size` slices using `iloc`, processes each chunk separately, then concatenates all results at the end.

Chunk Processing

Applies a custom calculation per value, then uses `np.where` to log-transform positive results and set zeros for non-positive ones.

**Performance Tip**: Use chunking for large datasets that don't fit in memory!

***

### Working with Time Series Data

Efficient time series processing:

Import

Only pandas is needed here; NumPy is accessed via the module-level `np` alias for the trend comparison.

Rolling Window Analysis

Sorts by date, computes 7-day rolling mean and standard deviation, then classifies each point as Upward or Downward by comparing the mean to its previous value.

Group Statistics

Applies an inner function to each group that returns a Series of summary stats, mean, std, count, and an outlier flag, via `groupby.apply`.

## Common Data Processing Patterns

***

### Pattern: Data Validation

Common validation patterns:

Validator Init

Stores the DataFrame and an empty list for accumulating issue messages so all checks can be batched before reporting.

Range Validation

Uses `between` to create a boolean mask, inverts it to find out-of-range rows, and appends a message only when violations exist.

Category Validation and Report

Checks that all values are in the allowed set using `isin`, then `get_validation_report` joins all collected messages or returns a pass confirmation.

***

### Pattern: Data Cleaning

Standard cleaning operations:

Safe Copy Init

Makes a copy of the DataFrame at construction so the original is never mutated by the cleaning methods.

Numeric Cleaning

Coerces non-numeric strings to NaN with `pd.to_numeric`, then nulls out values whose Z-score exceeds 3 standard deviations.

Categorical Cleaning

Lowercases and strips whitespace for consistency, then collapses infrequent categories (fewer than 10 rows) into `'other'` to reduce cardinality.

## Practice Exercises

> **Pro Tip:** Start with simple examples in Python Tutor, then scale up in Colab!

### Exercise 1: Grade Calculator

```python
def assign_grade(score):
   """
   Assign letter grade based on score:
   90-100: A
   80-89: B
   70-79: C
   60-69: D
   Below 60: F
   """
   # Your code here...
   pass

# Test with multiple scores
scores = [95, 87, 73, 62, 58, 91]
for score in scores:
   grade = assign_grade(score)
   print(f"Score {score}: Grade {grade}")
```

```
Score 95: Grade None
Score 87: Grade None
Score 73: Grade None
Score 62: Grade None
Score 58: Grade None
Score 91: Grade None
```

> **Visualize:** Paste into Python Tutor to see the loop iterate! **Ask:** "What are best practices for grade boundaries in code?"

### Exercise 2: Data Validator with Loops

```python
def validate_dataset(data):
   """
   Check each value in data:
   - Flag if negative
   - Flag if above 1000
   - Count valid values
   - Return report
   """
   report = {
       'total': 0,
       'valid': 0,
       'negative': 0,
       'too_high': 0
   }

   # Your code here...

   return report

# Test it:
test_data = [10, -5, 50, 1200, 30, -10, 800, 45]
result = validate_dataset(test_data)
print(result)
```

```
{'total': 0, 'valid': 0, 'negative': 0, 'too_high': 0}
```

> **Watch Counters:** Python Tutor shows how report values update in the loop! **Improve:** "Suggest ways to make this validation function more reliable"

### Exercise 3: Nested Loops for Matrix Operations

```python
def process_matrix(matrix):
   """
   Process a 2D matrix:
   - Find max value in each row
   - Calculate sum of each column
   - Find overall maximum
   """
   # Example matrix:
   # [[1, 2, 3],
   #  [4, 5, 6],
   #  [7, 8, 9]]

   # Your code here...
   pass

# Test it:
matrix = [
   [10, 20, 30],
   [40, 50, 60],
   [70, 80, 90]
]
result = process_matrix(matrix)
```

> **Nested Loop Visualization:** Python Tutor makes nested loops crystal clear! **Challenge:** "Show me how to do this with numpy instead of loops"

### Exercise 4: While Loop for Data Processing

```python
def process_until_threshold(data, threshold=100):
   """
   Process data items until sum reaches threshold:
   - Add values one by one
   - Stop when threshold reached
   - Return: items used, total, remaining items
   """
   # Your code here...
   pass

# Test it:
values = [10, 20, 15, 30, 25, 40, 20]
used, total, remaining = process_until_threshold(values, threshold=100)
print(f"Used {used} items, Total: {total}, Remaining: {remaining}")
```

> **Safety First:** Python Tutor helps catch infinite loops before they crash! **Learn:** "When should I use while loops vs for loops?"

## Advanced Challenges

### Challenge 1: Fizz Buzz (Classic Interview Question!)

```python
# Print numbers 1-100, but:
# - "Fizz" for multiples of 3
# - "Buzz" for multiples of 5
# - "FizzBuzz" for multiples of both
# Your code here...
```

### Challenge 2: Pattern Matching

Create a function that finds patterns in data sequences.

### Challenge 3: Data Grouping

Group data into categories based on multiple conditions.

> **Video Help:** [Video Resources](video-resources.md) - Loops & Conditions section

## Common Mistakes & Debugging

### Mistake 1: Infinite Loops

```python
# Wrong (avoid):
i = 0
while i < 5:
   print(i)
   # Forgot to increment!

# Right (preferred):
i = 0
while i < 5:
   print(i)
   i += 1
```

> **Catch It Early:** Python Tutor shows you're stuck before you crash!

### Mistake 2: Off-by-One Errors

```python
# Common mistake
numbers = [10, 20, 30, 40, 50]

# Wrong (avoid):
for i in range(len(numbers)):
   print(numbers[i + 1])  # Will crash!

# Right (preferred):
for i in range(len(numbers)):
   print(numbers[i])

# Better (preferred):
for number in numbers:
   print(number)
```

> **See The Error:** Python Tutor shows exactly where index goes out of bounds!

### Mistake 3: Modifying List While Iterating

```python
# Wrong (avoid):
numbers = [1, 2, 3, 4, 5]
for num in numbers:
   if num % 2 == 0:
       numbers.remove(num)  # Dangerous!

# Right (preferred):
numbers = [1, 2, 3, 4, 5]
numbers = [num for num in numbers if num % 2 != 0]
```

> **Debug:** "Why shouldn't I modify a list while iterating over it?"

Remember:

* Use vectorized operations when possible
* Consider memory efficiency
* Handle edge cases
* Validate results
* **Visualize loops in Python Tutor to understand flow**
* **Use AI to debug logical errors**
* **Test with edge cases (empty lists, single items, etc.)**

## Common pitfalls

* **Off-by-one errors**: Check whether your range includes the last index; **range(len(x))** vs **range(len(x)-1)** trips people up.
* **Modifying a list while iterating**: Prefer building a new list or iterate over a copy.
* **Infinite loops**: Ensure the condition can become false (especially with **while**).

## Next steps

Continue to [Functions](functions.md) to package logic into reusable pieces.

Happy analyzing!
