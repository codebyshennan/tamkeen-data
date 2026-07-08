# Function Application and Mapping in Pandas

**After this lesson:** you can explain Function Application and Mapping in Pandas and try the examples in your own notebook.

### Video

_Corey Schafer, Python pandas tutorial (part 5): updating rows and columns (maps, replaces, transforms)_

## Overview

**Prerequisites:** Python **functions** and [Series](series.md) / [DataFrame](dataframe.md) basics.

**Why this lesson:** Real tables need row- or column-level rules: normalize strings, parse dates, clip outliers, or compute features. **apply**, **map**, **applymap** (older) / **map** + **assign**, and **transform** are the main ways to plug Python logic into pandas, knowing _when_ each is appropriate keeps code fast and readable.

## Understanding Function Application

***

### What is Function Application?

Function application in Pandas means applying a function to your data to transform or analyze it. Think of it like:

_`apply` is flexible but slow on large DataFrames, prefer built-in vectorised methods (`.str`, `.dt`, arithmetic) whenever they exist._

* A recipe that you apply to each ingredient
* A rule that processes each piece of data
* A transformation that changes values
* A filter that selects specific data
* An analysis that extracts insights

Key benefits:

* Efficient data processing
* Consistent transformations
* Complex calculations
* Custom data manipulation

Real-world applications:

* Financial calculations (interest, tax)
* Data cleaning (standardization)
* Feature engineering (ML preparation)
* Date/time processing
* Text analysis and cleaning

***

### Basic Function Application

we will look at with practical examples:

**`Series.apply` for grades and cleaning**

* **Purpose:** Map numeric scores to letter buckets with a Python function, then clean currency strings and whitespace on a second table.
* **Walkthrough:** `df[subject].apply(to_letter_grade)` is per-cell; `clean_price` strips `$` and commas before `float`.

```
Original grades:
      Name  Math  Science  History
0    Alice    85       92       88
1      Bob    76       88       82
2  Charlie    92       95       85

With letter grades:
      Name  Math  Science  History Math_Grade Science_Grade History_Grade
0    Alice    85       92       88          B             A             B
1      Bob    76       88       82          C             B             B
2  Charlie    92       95       85          A             A             B

Cleaned sales data:
         Date   Product    Price  Quantity    Total
0  2023-01-01    Laptop  1200.00         5  6000.00
1  2023-01-02     Mouse    25.99        10   259.90
2  2023-01-03  Keyboard    89.99         8   719.92
```

Imports

Imports pandas and NumPy, the only two dependencies for basic function application examples.

Grade Bucketing

Defines a letter-grade function and applies it to each numeric column with a loop, the simplest per-element apply pattern.

Data Cleaning

Defines separate clean functions for price strings and product names, then chains `apply`, `astype`, and arithmetic to produce a tidy sales total column.

```
Original grades:
      Name  Math  Science  History
0    Alice    85       92       88
1      Bob    76       88       82
2  Charlie    92       95       85

With letter grades:
      Name  Math  Science  History Math_Grade Science_Grade History_Grade
0    Alice    85       92       88          B             A             B
1      Bob    76       88       82          C             B             B
2  Charlie    92       95       85          A             A             B

Cleaned sales data:
         Date   Product    Price  Quantity    Total
0  2023-01-01    Laptop  1200.00         5  6000.00
1  2023-01-02     Mouse    25.99        10   259.90
2  2023-01-03  Keyboard    89.99         8   719.92
```

## Different Ways to Apply Functions

***

### Using apply()

The `apply()` method is the most versatile way to apply functions:

**Row-wise `apply` with `axis=1`**

* **Purpose:** Compute a value from multiple columns in the same row (revenue = price × quantity).
* **Walkthrough:** `calculate_revenue(row)` receives a Series for each row when `axis=1`.

```python
# Create a DataFrame with sales data
sales_df = pd.DataFrame({
    'Product': ['Apple', 'Banana', 'Orange'],
    'Price': [0.5, 0.3, 0.6],
    'Quantity': [100, 150, 80]
})

# Calculate total revenue for each product
def calculate_revenue(row):
    return row['Price'] * row['Quantity']

sales_df['Revenue'] = sales_df.apply(calculate_revenue, axis=1)

print("Sales with revenue:")
print(sales_df)
```

```
Sales with revenue:
  Product  Price  Quantity  Revenue
0   Apple    0.5       100     50.0
1  Banana    0.3       150     45.0
2  Orange    0.6        80     48.0
```

The **axis** parameter determines whether the function is applied to:

* rows (**axis=1**)
* columns (**axis=0** or default)

***

### Using applymap()

`applymap()` applies a function to every single element in a DataFrame:

**Element-wise formatting (use `map` in modern pandas for a single column)**

* **Purpose:** Transform every cell, here format floats as strings with two decimals for display.
* **Walkthrough:** `applymap` receives scalar `x` per cell; pandas 2.1+ prefers `DataFrame.map` for the same idea.

```python
# Create a DataFrame with numbers
df = pd.DataFrame({
    'A': [1.23456, 2.34567, 3.45678],
    'B': [4.56789, 5.67890, 6.78901]
})

# Format all numbers to 2 decimal places
formatted_df = df.applymap(lambda x: f"{x:.2f}")

print("Original numbers:")
print(df)
print("\nFormatted numbers:")
print(formatted_df)
```

***

### Using map() with Series

For Series objects, use `map()` to transform values:

**Dictionary lookup on a Series**

* **Purpose:** Replace codes with human-readable labels using a dict, faster than nested `if` for large code tables.
* **Walkthrough:** `products.map(product_names)` aligns index to the original Series.

```python
# Create a Series of product codes
products = pd.Series(['A123', 'B456', 'C789'])

# Create a mapping dictionary
product_names = {
    'A123': 'Laptop',
    'B456': 'Mouse',
    'C789': 'Keyboard'
}

# Map codes to names
product_labels = products.map(product_names)

print("Product codes:")
print(products)
print("\nProduct names:")
print(product_labels)
```

```
Product codes:
0    A123
1    B456
2    C789
dtype: str

Product names:
0      Laptop
1       Mouse
2    Keyboard
dtype: str
```

## Real-World Examples

***

### Data Cleaning Example

Clean and standardize customer data:

**Title-case names and normalize phones**

* **Purpose:** Show per-column cleaning pipelines, string methods for names, digit extraction for phones.
* **Walkthrough:** `filter(str.isdigit, phone)` keeps digits before slicing into `XXX-XXX-XXXX`.

```python
# Create a DataFrame with messy customer data
customers = pd.DataFrame({
    'Name': ['john doe', 'JANE SMITH', 'Bob Wilson'],
    'Email': ['john@email.com', 'jane@email.com', 'bob@email.com'],
    'Phone': ['123-456-7890', '(987) 654-3210', '555.444.3333']
})

# Clean up names
def clean_name(name):
    return name.title()  # Capitalize first letter of each word

# Standardize phone numbers
def standardize_phone(phone):
    # Remove all non-numeric characters
    numbers_only = ''.join(filter(str.isdigit, phone))
    # Format as XXX-XXX-XXXX
    return f"{numbers_only[:3]}-{numbers_only[3:6]}-{numbers_only[6:]}"

# Apply cleaning functions
customers['Name'] = customers['Name'].apply(clean_name)
customers['Phone'] = customers['Phone'].apply(standardize_phone)

print("Cleaned customer data:")
print(customers)
```

```
Cleaned customer data:
         Name           Email         Phone
0    John Doe  john@email.com  123-456-7890
1  Jane Smith  jane@email.com  987-654-3210
2  Bob Wilson   bob@email.com  555-444-3333
```

***

### Data Analysis Example

Calculate statistics for student grades:

**Return a Series per row from `apply`**

* **Purpose:** Aggregate each student's quiz columns into average, min, max, and a boolean trend flag in one pass.
* **Walkthrough:** `analyze_grades` returns `pd.Series(...)`; `apply(..., axis=1)` stacks those into new columns.

\`\`\` Original grades: Student Quiz1 Quiz2 Quiz3 0 Alice 95 88 92 1 Bob 80 85 88 2 Charlie 85 90 85 3 David 70 75 80

Grade analysis: Average Highest Lowest Improved 0 91.666667 95 88 False 1 84.333333 88 80 True 2 86.666667 90 85 False 3 75.000000 80 70 True

```


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Quiz DataFrame</span>
    </div>
    <div class="code-callout__body">
      <p>Creates a small four-student quiz DataFrame to illustrate how row-wise <code>apply</code> can produce multiple derived columns at once.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="9-17" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Multi-Column Return</span>
    </div>
    <div class="code-callout__body">
      <p>Returns a <code>pd.Series</code> with four metrics per row, mean, max, min, and an improvement boolean, so <code>apply</code> expands them into columns automatically.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="19-24" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Row-Wise Apply</span>
    </div>
    <div class="code-callout__body">
      <p>Calls <code>apply(..., axis=1)</code> to run the function once per row and prints both the original table and the new analysis side by side.</p>
    </div>
  </div>
</aside>
</div>

```

Original grades: Student Quiz1 Quiz2 Quiz3 0 Alice 95 88 92 1 Bob 80 85 88 2 Charlie 85 90 85 3 David 70 75 80

Grade analysis: Average Highest Lowest Improved 0 91.666667 95 88 False 1 84.333333 88 80 True 2 86.666667 90 85 False 3 75.000000 80 70 True

````

## Best Practices and Tips

---

### Performance Considerations

1. **Vectorization Over Iteration**:

   ```python
   # Slower: Using apply for simple operations
   df['Double'] = df['Value'].apply(lambda x: x * 2)

   # Faster: Using vectorized operations
   df['Double'] = df['Value'] * 2
````

2.  **Built-in Methods**:

    ```python
    # Use built-in methods when available
    # Instead of:
    df['Sum'] = df[['A', 'B']].apply(lambda x: x['A'] + x['B'], axis=1)

    # Use:
    df['Sum'] = df['A'] + df['B']
    ```

***

### Function Design Tips

1.  **Keep Functions Simple**:

    ```python
    # Good: Single responsibility
    def calculate_tax(amount):
        return amount * 0.2

    # Bad: Too many responsibilities
    def process_sale(amount):
        tax = amount * 0.2
        shipping = 10
        discount = amount * 0.1
        return amount + tax + shipping - discount
    ```
2.  **Handle Edge Cases**:

    ```python
    def safe_division(x):
        try:
            return 100 / x
        except ZeroDivisionError:
            return np.nan

    df['Result'] = df['Value'].apply(safe_division)
    ```

## Common Pitfalls and Solutions

1.  **Modifying Data During Apply**:

    ```python
    # Wrong: Modifying DataFrame during apply
    def bad_function(row):
        df.at[row.name, 'NewCol'] = row['Value'] * 2  # Don't do this
        return row

    # Right: Return new values
    def good_function(row):
        return row['Value'] * 2
    ```
2.  **Choosing the Wrong Axis**:

    ```python
    # Remember:
    # axis=0 (default) -> apply function to each column
    # axis=1 -> apply function to each row

    # For row operations:
    df.apply(func, axis=1)

    # For column operations:
    df.apply(func, axis=0)  # or just df.apply(func)
    ```
3.  **Performance with Large Datasets**:

    ```python
    # If possible, use vectorized operations
    # Instead of:
    df['Celsius'] = df['Fahrenheit'].apply(lambda x: (x - 32) * 5/9)

    # Use:
    df['Celsius'] = (df['Fahrenheit'] - 32) * 5/9
    ```

Remember: Choose the right function application method based on your needs, and always consider performance implications when working with large datasets!

## Next steps

Continue to [Sorting and ranking](sorting-ranking.md), then [Arithmetic and alignment](arithmetic-alignment.md).
