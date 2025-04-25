# Function Application and Mapping in Pandas

## Understanding Function Application

{% stepper %}
{% step %}

### What is Function Application?

Function application in Pandas means applying a function to your data to transform or analyze it. Think of it like:

- A recipe that you apply to each ingredient
- A rule that processes each piece of data
- A transformation that changes values
- A filter that selects specific data
- An analysis that extracts insights

Key benefits:

- Efficient data processing
- Consistent transformations
- Complex calculations
- Custom data manipulation

Real-world applications:

- Financial calculations (interest, tax)
- Data cleaning (standardization)
- Feature engineering (ML preparation)
- Date/time processing
- Text analysis and cleaning
{% endstep %}

{% step %}

### Basic Function Application

Let's explore with practical examples:

```python
import pandas as pd
import numpy as np

# Example 1: Student Grades Processing
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Math': [85, 76, 92],
    'Science': [92, 88, 95],
    'History': [88, 82, 85]
})

print("Original grades:")
print(df)

# Convert numerical grades to letter grades
def to_letter_grade(score):
    if score >= 90: return 'A'
    elif score >= 80: return 'B'
    elif score >= 70: return 'C'
    else: return 'F'

# Apply to all numeric columns
for subject in ['Math', 'Science', 'History']:
    df[f'{subject}_Grade'] = df[subject].apply(to_letter_grade)

print("\nWith letter grades:")
print(df)

# Example 2: Data Cleaning and Transformation
sales_data = pd.DataFrame({
    'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'Product': ['Laptop', 'Mouse   ', ' Keyboard'],
    'Price': ['$1,200', '$25.99', '$89.99'],
    'Quantity': ['5', '10', '8']
})

# Clean and transform data
def clean_price(price):
    return float(price.replace('$', '').replace(',', ''))

def clean_product(product):
    return product.strip()

sales_data['Price'] = sales_data['Price'].apply(clean_price)
sales_data['Product'] = sales_data['Product'].apply(clean_product)
sales_data['Quantity'] = sales_data['Quantity'].astype(int)
sales_data['Total'] = sales_data['Price'] * sales_data['Quantity']

print("\nCleaned sales data:")
print(sales_data)
```

{% endstep %}
{% endstepper %}

## Different Ways to Apply Functions

{% stepper %}
{% step %}

### Using apply()

The `apply()` method is the most versatile way to apply functions:

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

The `axis` parameter determines whether the function is applied to:

- rows (`axis=1`)
- columns (`axis=0` or default)
{% endstep %}

{% step %}

### Using applymap()

`applymap()` applies a function to every single element in a DataFrame:

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

{% endstep %}

{% step %}

### Using map() with Series

For Series objects, use `map()` to transform values:

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

{% endstep %}
{% endstepper %}

## Real-World Examples

{% stepper %}
{% step %}

### Data Cleaning Example

Clean and standardize customer data:

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

{% endstep %}

{% step %}

### Data Analysis Example

Calculate statistics for student grades:

```python
# Create grade data
grades = pd.DataFrame({
    'Student': ['Alice', 'Bob', 'Charlie', 'David'],
    'Quiz1': [95, 80, 85, 70],
    'Quiz2': [88, 85, 90, 75],
    'Quiz3': [92, 88, 85, 80]
})

# Calculate various statistics
def analyze_grades(row):
    grades_only = row[['Quiz1', 'Quiz2', 'Quiz3']]
    return pd.Series({
        'Average': grades_only.mean(),
        'Highest': grades_only.max(),
        'Lowest': grades_only.min(),
        'Improved': grades_only['Quiz3'] > grades_only['Quiz1']
    })

# Apply analysis
analysis = grades.apply(analyze_grades, axis=1)

print("Original grades:")
print(grades)
print("\nGrade analysis:")
print(analysis)
```

{% endstep %}
{% endstepper %}

## Best Practices and Tips

{% stepper %}
{% step %}

### Performance Considerations

1. **Vectorization Over Iteration**:

   ```python
   # Slower: Using apply for simple operations
   df['Double'] = df['Value'].apply(lambda x: x * 2)
   
   # Faster: Using vectorized operations
   df['Double'] = df['Value'] * 2
   ```

2. **Built-in Methods**:

   ```python
   # Use built-in methods when available
   # Instead of:
   df['Sum'] = df[['A', 'B']].apply(lambda x: x['A'] + x['B'], axis=1)
   
   # Use:
   df['Sum'] = df['A'] + df['B']
   ```

{% endstep %}

{% step %}

### Function Design Tips

1. **Keep Functions Simple**:

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

2. **Handle Edge Cases**:

   ```python
   def safe_division(x):
       try:
           return 100 / x
       except ZeroDivisionError:
           return np.nan
   
   df['Result'] = df['Value'].apply(safe_division)
   ```

{% endstep %}
{% endstepper %}

## Common Pitfalls and Solutions

1. **Modifying Data During Apply**:

   ```python
   # Wrong: Modifying DataFrame during apply
   def bad_function(row):
       df.at[row.name, 'NewCol'] = row['Value'] * 2  # Don't do this
       return row
   
   # Right: Return new values
   def good_function(row):
       return row['Value'] * 2
   ```

2. **Choosing the Wrong Axis**:

   ```python
   # Remember:
   # axis=0 (default) -> apply function to each column
   # axis=1 -> apply function to each row
   
   # For row operations:
   df.apply(func, axis=1)
   
   # For column operations:
   df.apply(func, axis=0)  # or just df.apply(func)
   ```

3. **Performance with Large Datasets**:

   ```python
   # If possible, use vectorized operations
   # Instead of:
   df['Celsius'] = df['Fahrenheit'].apply(lambda x: (x - 32) * 5/9)
   
   # Use:
   df['Celsius'] = (df['Fahrenheit'] - 32) * 5/9
   ```

Remember: Choose the right function application method based on your needs, and always consider performance implications when working with large datasets!
