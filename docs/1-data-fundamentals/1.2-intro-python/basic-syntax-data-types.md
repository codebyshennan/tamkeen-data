# basic-syntax-data-types

## Basic Syntax and Data Types for Data Science

**After this lesson:** You can write small Python programs using variables, core types, operators, and formatted output, enough to read and tweak data-science examples.

### Overview

**Prerequisites:** [Introduction to Python](./) module context; no other programming background required.

**Why this lesson:** Variables, operators, types, and f-strings are the **grammar** of every script you will read. Nothing here is "data science-only," but skipping it makes later pandas errors impossible to debug.

> **AI learning tip:** As you go through this section, try asking ChatGPT or Claude: "Explain Python variables using everyday objects as examples"

> **Visualize execution:** Open [Python Tutor](https://pythontutor.com) in another tab. Paste every code example to see exactly how Python executes it.

> **Interactive notebook:** [Open in Google Colab](notebooks/01-basic-syntax.ipynb), run and modify examples interactively.

#### Video

_Corey Schafer, Integers, floats, and numeric types in Python_

### Getting Started with Python

***

#### Your First Data Analysis Program

Start with a simple data analysis example:

**Mean and standard deviation of a small list**

* **Purpose:** Connect imports to a one-line summary: central tendency (`np.mean`) and spread (`np.std`) with readable f-string output.
* **Walkthrough:** `data` is a plain Python list; NumPy functions reduce it to scalars printed below the code.

```python
# Import essential libraries
import pandas as pd
import numpy as np

# Create some sample data
data = [10, 15, 20, 25, 30]

# Calculate basic statistics
mean = np.mean(data)
std = np.std(data)

print(f"Data Analysis Results:")
print(f"Mean: {mean}")
print(f"Standard Deviation: {std}")
```

```
Data Analysis Results:
Mean: 20.0
Standard Deviation: 7.0710678118654755
```

This example demonstrates:

* Importing libraries (**import**statement)
* Creating a data list
* Using functions for analysis
* Formatted output with f-strings

> **Visualize This in Python Tutor:**
>
> 1. Go to [pythontutor.com](https://pythontutor.com)
> 2. Paste the code above
> 3. Click "Visualize Execution"
> 4. Step through to see how imports work and functions execute

> **Try These AI Prompts:**
>
> * "Explain what `import` does in Python using a library analogy"
> * "Show me 3 different ways to print formatted output in Python"
> * "What happens in memory when I import a library?"

***

#### Running Python Code for Data Analysis

Two main approaches for data analysis:

1.  **Interactive Mode (Jupyter Notebook)**:

    **Inline CSV string to DataFrame**

    * **Purpose:** Mimic reading a file with `StringIO` so the snippet runs without an external CSV; `head()` previews rows in a notebook.
    * **Walkthrough:** `pd.read_csv(StringIO(sales_csv))` parses the triple-quoted string as file contents.

    ```python
    ```

## In Jupyter cell

import pandas as pd from io import StringIO

## Example rows (in a real notebook you would use pd.read\_csv("sales\_data.csv"))

sales\_csv = """amount 100 200 150""" df = pd.read\_csv(StringIO(sales\_csv)) df.head()

```
```

amount 0 100 1 200 2 150

````

  Perfect for exploratory data analysis!

2. **Script Mode (Production Code)**:

   **Reusable function + `if __name__`-style flow**

   - **Purpose:** Encapsulate loading and aggregation in `analyze_sales` so the same logic can target a file path or a `StringIO` buffer.
   - **Walkthrough:** The dict returned uses column sums/means from pandas-`StringIO` again stands in for a real path.

   ```python
  # analysis.py
  import pandas as pd
  import numpy as np
  from io import StringIO

  def analyze_sales(source):
      df = pd.read_csv(source)
      return {
          'total_sales': df['amount'].sum(),
          'average_sale': df['amount'].mean()
      }

  sales_csv = """amount
  100
  200
  150"""
  results = analyze_sales(StringIO(sales_csv))
  print(results)
````

```
{'total_sales': np.int64(450), 'average_sale': np.float64(150.0)}
```

### Python Syntax for Data Analysis

***

#### Indentation in Data Processing

Python's indentation is important in data processing flows:

**Nested blocks: function → if → for**

* **Purpose:** See how indentation defines scope-`cleaned_data` is built only when `len(data) > 0` and each value passes `pd.notna`.
* **Walkthrough:** `pd.notna(value)` filters out `None` and NumPy NaN in a list context.

```python
def process_data(data):
   # First level: Function body
   if len(data) > 0:
       # Second level: Inside if statement
       cleaned_data = []
       for value in data:
           # Third level: Inside loop
           if pd.notna(value):  # Check for non-NA values
               cleaned_data.append(value)

   return cleaned_data

# Example usage
raw_data = [10, None, 20, np.nan, 30]
clean_data = process_data(raw_data)
```

**Pro Tip**: Consistent indentation is important for maintaining complex data processing pipelines.

> **See Indentation in Action:** Paste this code into Python Tutor to see how indentation creates code blocks:
>
> ```python
> x = 5
> if x > 0:
>   print("Positive")
>   print("Still inside if")
> print("Outside if")
> ```

```
Positive
Still inside if
Outside if
```

> **Debug with AI:** If you get an `IndentationError`, paste your code into ChatGPT and ask: "Fix the indentation in this Python code: \[paste code]"

***

#### Comments in Data Analysis Code

Good documentation is essential in data science:

**Docstring + sklearn `StandardScaler` on numeric columns**

* **Purpose:** Show a typical preprocessing skeleton: drop NA, then scale only numeric columns selected by `select_dtypes`.
* **Walkthrough:** `scaler.fit_transform` returns an ndarray assigned back to `df[numeric_cols]`.

Imports

Imports pandas, NumPy, and sklearn's `StandardScaler`-the three tools needed for this preprocessing pipeline.

Docstring

A full docstring documents the parameter type and return type, good practice so callers know what to pass and what comes back.

Clean and Scale

Drops NaN rows first, then uses `select_dtypes` to find numeric columns and scales them with `StandardScaler`-a common two-step before model training.

**Best Practice**: Use docstrings for functions and detailed inline comments for complex operations.

### Variables and Data Types in Data Science

***

#### Variables in Data Analysis

Variables in data science often represent different types of data:

**Representing common data-science dtypes**

* **Purpose:** Tie variable names to roles: continuous vs discrete numbers, nominal vs ordinal categories, datetimes, and small `DataFrame` tables.
* **Walkthrough:** `np.array` vs `pd.DataFrame` previews Module 1.4-1.5 content.

```python
# Numerical data
temperature = 23.5     # Continuous data
count_users = 1000     # Discrete data

# Categorical data
category = "Electronics"  # Nominal data
rating = "A"        # Ordinal data

# Time-based data
from datetime import datetime
timestamp = datetime.now() # Time data

# Arrays and matrices
import numpy as np
data_array = np.array([1, 2, 3, 4, 5])
data_matrix = np.array([[1, 2], [3, 4]])

# DataFrame
import pandas as pd
df = pd.DataFrame({
   'id': range(1, 4),
   'value': [10, 20, 30]
})
```

**Remember**: Choose appropriate data types for efficient memory usage and processing!

> **Experiment in Python Tutor:** See how different variable types are stored in memory:
>
> ```python
> number = 42
> decimal = 3.14
> text = "Hello"
> flag = True
> # Watch how Python allocates memory for each type
> ```

> **AI Learning Exercise:** Ask: "Create a quiz with 5 questions about Python data types for beginners"

***

#### Variable Naming in Data Science

Follow these conventions for clear data analysis code:

**Do This**:

**Clear names for analysis variables**

* **Purpose:** Contrast readable names with vague ones, same values, better scan-ability in notebooks and reviews.

```python
mean_temperature = 23.5   # Clear statistical measure
customer_id = "C001"    # Entity identifier
is_outlier = True     # Boolean flag
daily_sales_data = df   # DataFrame content
MAX_ITERATIONS = 1000   # Constant value
```

**Don't Do This**:

**Avoid vague identifiers**

* **Purpose:** See how `temp`, `data1`, `x` obscure intent compared to the "Do This" block above, no behavior change, only naming.

```python
temp = 23.5        # Too vague
data1 = pd.DataFrame()   # Uninformative name
x = np.array([1,2,3])   # Unclear purpose
```

**Pro Tip**: Use descriptive names that indicate the variable's role in your analysis.

***

#### Data Types for Analysis

Python data types commonly used in data science:

_When you see a pandas `dtype` of `object`, it usually means string. `float64` columns often signal that pandas introduced NaN (which forces float) to hold missing values._

1.  **Numeric Types for Statistical Analysis**:

    ```python
    ```

import numpy as np

## Integer types

sample\_size = 1000 # int array\_int = np.int32(\[1, 2, 3]) # numpy int32

## Float types

mean\_value = 75.5 # float array\_float = np.float64(\[1.1, 1.2]) # numpy float64

## Complex numbers (e.g., for signal processing)

signal = 3 + 4j

````

2. **Text Data for Natural Language Processing**:

```python
# String operations for text analysis
text = "Data Science is fascinating"
tokens = text.lower().split()

# Regular expressions for pattern matching
import re
emails = re.findall(r'\S+@\S+', text)
````

3.  **Boolean Arrays for Filtering**:

    ```python
    ```

import pandas as pd

df = pd.DataFrame({ 'value': \[10, 20, 30, 40, 50] })

## Boolean indexing

mask = df\['value'] > 30 high\_values = df\[mask]

````

4. **Special Types for Missing Data**:

```python
# None for missing values
optional_value = None

# NaN for numerical missing data
missing_numeric = np.nan

# Handling missing data in pandas
df = pd.DataFrame({
   'value': [10, np.nan, 30]
})
clean_df = df.dropna()
````

**Tip**: Use `dtype` to check array types in NumPy/Pandas:

```python
print(df['value'].dtype) # dtype('float64')
print(np.array([1, 2]).dtype) # dtype('int64')
```

```
int64
int64
```

### Working with Numbers in Data Analysis

***

#### Mathematical Operations for Data Science

Common numerical operations in data analysis:

```python
import numpy as np

# Basic statistics
data = [1, 2, 3, 4, 5]
mean = np.mean(data)    # 3.0
median = np.median(data)  # 3.0
std = np.std(data)     # Standard deviation

# Matrix operations
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])

# Matrix multiplication
product = np.dot(matrix_a, matrix_b)

# Element-wise operations
sum_matrix = matrix_a + matrix_b
diff_matrix = matrix_a - matrix_b

# Statistical functions
correlation = np.corrcoef(data, data)
```

**Pro Tip**: Use NumPy for efficient numerical computations with large datasets!

***

#### Numerical Precision and Types

Understanding precision in data analysis:

```python
import numpy as np

# Integer precision
int32_array = np.array([1, 2, 3], dtype=np.int32)
int64_array = np.array([1, 2, 3], dtype=np.int64)

# Float precision
float32_array = np.array([1.1, 1.2, 1.3], dtype=np.float32)
float64_array = np.array([1.1, 1.2, 1.3], dtype=np.float64)

# Memory usage
print(f"Int32 memory: {int32_array.nbytes} bytes")
print(f"Float64 memory: {float64_array.nbytes} bytes")

# Precision considerations
a = 0.1 + 0.2
b = 0.3
print(f"0.1 + 0.2 == 0.3: {abs(a - b) < 1e-10}") # Use tolerance for float comparison
```

```
Int32 memory: 12 bytes
Float64 memory: 24 bytes
0.1 + 0.2 == 0.3: True
```

### String Operations in Data Analysis

***

#### Text Data Processing

Common string operations in data analysis:

```python
# Text cleaning
text = " Data Science "
cleaned = text.strip().lower() # Remove whitespace and convert to lowercase

# Pattern matching
import re
text = "Temperature: 23.5°C"
temperature = float(re.findall(r'\d+\.\d+', text)[0])

# String parsing for data extraction
date_str = "2023-01-01"
from datetime import datetime
date_obj = datetime.strptime(date_str, '%Y-%m-%d')

# Working with CSV data
csv_line = "id,name,value"
columns = csv_line.split(',')
```

***

#### String Formatting in Reports

Format strings for data reporting:

```python
# Formatting numerical results
accuracy = 0.9567
print(f"Model Accuracy: {accuracy:.2%}") # 95.67%

# Table-like output
data = {
   'precision': 0.95,
   'recall': 0.92,
   'f1_score': 0.93
}

# Create formatted report
report = """
Model Metrics:
-------------
Precision: {precision:.2%}
Recall: {recall:.2%}
F1 Score: {f1_score:.2%}
""".format(**data)

print(report)
```

```
Model Accuracy: 95.67%

Model Metrics:
-------------
Precision: 95.00%
Recall: 92.00%
F1 Score: 93.00%
```

### Type Conversion in Data Processing

***

#### Data Type Conversions

Common type conversions in data analysis:

_Always validate before converting, a stray `"N/A"` string will raise a `ValueError` if you call `int()` directly. Use `pd.to_numeric(errors='coerce')` to convert safely and let pandas turn failures into `NaN`._

```python
import pandas as pd
import numpy as np

# Converting strings to numbers
numeric_strings = ['1', '2.5', '3.14']
integers = [int(x) for x in numeric_strings if '.' not in x]
floats = [float(x) for x in numeric_strings]

# Converting to categorical
categories = pd.Categorical(['A', 'B', 'A', 'C'])
encoded = pd.get_dummies(categories)

# DateTime conversions
dates = ['2023-01-01', '2023-01-02']
datetime_objects = pd.to_datetime(dates)

# Array type conversion
float_array = np.array([1, 2, 3], dtype=float)
int_array = float_array.astype(int)
```

**Warning**: Always validate data before conversion:

```python
def safe_float_convert(value):
   try:
       return float(value)
   except (ValueError, TypeError):
       return np.nan
```

***

#### Data Type Checking and Validation

Best practices for type checking in data analysis:

Imports

Only pandas and NumPy are needed, pandas for type selection, NumPy for the `isinf` check.

Numeric Checks

Loops over numeric columns to flag infinite values and negative numbers in columns whose names suggest they should be positive, two common data quality traps.

Categorical Checks

Counts unique values per object column and warns if cardinality exceeds 100, high-cardinality categoricals often indicate data entry errors or IDs mistaken for categories.

### Practice Exercises for Data Science

> **Pro Tip:** Try solving these in Google Colab first, then use Python Tutor to understand your solution!

Try these data analysis exercises:

#### Exercise 1: Temperature Analysis

* **Purpose:** Stub for **Exercise 1**-complete the bullets in the heading (stats + °F + outliers).

```python
# Create a numpy array of temperatures and calculate:
# - Mean, median, and standard deviation
# - Convert Celsius to Fahrenheit
# - Find outliers (values > 2 standard deviations)

# Start here:
import numpy as np
temperatures_celsius = np.array([22, 25, 19, 100, 23, 21, 24])
# Your code here...
```

> **Get Help:** If stuck, ask AI: "Walk me through solving this temperature analysis problem step by step"

#### Exercise 2: String Processing

* **Purpose:** Stub for **Exercise 2**-split, parse floats, summarize.

```python
# Process a string of comma-separated values:
# - Split into individual values
# - Convert numeric strings to floats
# - Calculate summary statistics

data_string = "10.5, 20.3, 15.7, 18.9, 22.1"
# Your code here...
```

> **Visualize It:** Use Python Tutor to see how string methods work

#### Exercise 3: Date Manipulation

* **Purpose:** Stub for **Exercise 3**-parse strings, differences, components.

```python
# Work with dates and times:
# - Convert string dates to datetime objects
# - Calculate time differences
# - Extract specific components (year, month, day)

from datetime import datetime
date_strings = ["2024-01-15", "2024-02-20", "2024-03-10"]
# Your code here...
```

#### Exercise 4: Data Cleaning Function

* **Purpose:** Stub for **Exercise 4**-implement `clean_data` per comments and test with `messy_data`.

```python
# Create a simple data cleaning function:
# - Remove missing values
# - Convert data types appropriately
# - Handle outliers

import numpy as np

def clean_data(data):
   # Your code here...
   pass

# Test with:
messy_data = [1, 2, None, 4, 100, 5, np.nan]
```

> **Code Review:** After solving, ask AI: "Review this code and suggest improvements: \[paste your solution]"

### Challenge Yourself

#### Beginner Challenge

Create a program that asks for your age and calculates how many days you've lived.

#### Intermediate Challenge

Build a temperature converter that handles Celsius, Fahrenheit, and Kelvin.

#### Advanced Challenge

Create a function that validates and cleans email addresses from a list.

> **Video Help:** Check our [Video Resources](video-resources.md) for tutorials on these topics!

Remember:

* Use numpy for numerical operations
* Pandas for structured data
* Always validate your data
* Handle errors gracefully
* **Use AI to explain concepts you don't understand**
* **Visualize confusing code in Python Tutor**

### Common pitfalls

* **Indentation errors**: Python uses indentation for blocks; mixing spaces and tabs breaks code. Use one style (usually four spaces) in your editor.
* **Name errors before assignment**: You must assign a variable before you use it; trace the order of lines when debugging.
* **Float comparison**: Avoid checking **==** between floats; use **math.isclose** or compare rounded values.

### Next steps

Continue to [Data structures](data-structures.md) for lists, dictionaries, tuples, and sets.

Happy analyzing!
