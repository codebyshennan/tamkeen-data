# Sorting and Ranking in Pandas

## Understanding Sorting

{% stepper %}
{% step %}
### What is Sorting?
Sorting in Pandas helps you organize your data in a specific order. Think of it like:
- 📚 Arranging books alphabetically on a shelf
- 📊 Organizing test scores from highest to lowest
- 📅 Arranging dates from oldest to newest
- 💰 Sorting transactions by amount
- 🏆 Ranking players by score

Key benefits:
- 🔍 Quick value lookup
- 📈 Pattern identification
- 📊 Data presentation
- 🎯 Priority identification
- 📉 Trend analysis

Real-world applications:
- 📈 Financial portfolio analysis
- 🏆 Sports rankings and statistics
- 📊 Sales performance reports
- 📅 Event scheduling and planning
- 🎯 Customer segmentation
{% endstep %}

{% step %}
### Basic Sorting Example
Let's explore sorting with practical examples:

```python
import pandas as pd
import numpy as np

# Example 1: Student Performance
scores = pd.Series({
    'Alice': 85,
    'Bob': 92,
    'Charlie': 78,
    'David': 95,
    'Eve': 88
}, name='Test Scores')

print("Original scores:")
print(scores)

# Sort by values (highest to lowest)
print("\nTop performers:")
print(scores.sort_values(ascending=False))

# Sort by student names
print("\nAlphabetical order:")
print(scores.sort_index())

# Example 2: Sales Analysis
sales_data = pd.DataFrame({
    'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Laptop'],
    'Price': [1200, 25, 100, 300, 1100],
    'Units': [5, 50, 30, 10, 8],
    'Date': pd.date_range('2023-01-01', periods=5)
})

# Calculate total sales
sales_data['Total'] = sales_data['Price'] * sales_data['Units']

print("\nSales Data (sorted by total sales):")
print(sales_data.sort_values('Total', ascending=False))
```
{% endstep %}
{% endstepper %}

## Sorting DataFrames

{% stepper %}
{% step %}
### Sorting by a Single Column
Let's work with a student grades DataFrame:

```python
# Create a DataFrame with student grades
grades = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Math': [85, 92, 78, 95, 88],
    'Science': [92, 85, 96, 88, 90],
    'History': [88, 85, 92, 85, 95]
})

print("Original grades:")
print(grades)

# Sort by Math scores
print("\nSorted by Math scores:")
print(grades.sort_values('Math'))

# Sort by Math scores in descending order
print("\nSorted by Math scores (highest to lowest):")
print(grades.sort_values('Math', ascending=False))
```
{% endstep %}

{% step %}
### Sorting by Multiple Columns
You can sort by multiple columns to break ties:

```python
# Sort by Science first, then by Math
print("Sorted by Science, then Math:")
print(grades.sort_values(['Science', 'Math']))

# Sort Science descending and Math ascending
print("\nScience descending, Math ascending:")
print(grades.sort_values(['Science', 'Math'], 
                        ascending=[False, True]))
```
{% endstep %}
{% endstepper %}

## Understanding Ranking

{% stepper %}
{% step %}
### What is Ranking?
Ranking assigns positions to your data based on their values. Think of it like:
- Ranking athletes in a competition
- Assigning class rank to students
- Determining the position of teams in a league

The difference from sorting is that ranking keeps your data in its original order but adds rank numbers.
{% endstep %}

{% step %}
### Basic Ranking Example
Let's see different ways to rank data:

```python
# Create a Series with test scores
scores = pd.Series([85, 92, 85, 95, 88])
print("Original scores:")
print(scores)

# Default ranking (average method for ties)
print("\nDefault ranking:")
print(scores.rank())

# Rank with different methods for handling ties
print("\nRank with 'first' method:")
print(scores.rank(method='first'))

print("\nRank with 'min' method:")
print(scores.rank(method='min'))

print("\nRank with 'max' method:")
print(scores.rank(method='max'))
```

Notice how different methods handle the tied scores (85 appears twice).
{% endstep %}
{% endstepper %}

## Real-World Examples

{% stepper %}
{% step %}
### Sales Performance Analysis
Let's analyze sales data:

```python
# Create sales data
sales = pd.DataFrame({
    'Salesperson': ['John', 'Sarah', 'Mike', 'Lisa', 'Tom'],
    'Region': ['North', 'South', 'North', 'South', 'North'],
    'Sales': [150000, 200000, 150000, 300000, 250000],
    'Clients': [50, 40, 45, 60, 55]
})

# Sort by sales and rank within regions
print("Sales data sorted by amount:")
print(sales.sort_values('Sales', ascending=False))

# Rank salespeople within their regions
sales['RegionalRank'] = sales.groupby('Region')['Sales'].rank(ascending=False)
print("\nSales with regional rankings:")
print(sales)
```
{% endstep %}

{% step %}
### Student Performance Analysis
Analyze student rankings across different subjects:

```python
# Create student data
students = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Math': [85, 92, 78, 95, 88],
    'Science': [92, 85, 96, 88, 90],
    'History': [88, 85, 92, 85, 95]
})

# Calculate overall ranking
students['AverageScore'] = students[['Math', 'Science', 'History']].mean(axis=1)
students['OverallRank'] = students['AverageScore'].rank(ascending=False)

# Calculate subject-wise rankings
for subject in ['Math', 'Science', 'History']:
    students[f'{subject}Rank'] = students[subject].rank(ascending=False)

print("Student rankings:")
print(students.sort_values('OverallRank'))
```
{% endstep %}
{% endstepper %}

## Best Practices and Tips

{% stepper %}
{% step %}
### Sorting Best Practices
1. **Preserve Original Data**:
   ```python
   # Create sorted view without modifying original
   sorted_df = df.sort_values('column')
   
   # Or sort in-place if needed
   df.sort_values('column', inplace=True)
   ```

2. **Handle Missing Values**:
   ```python
   # Control where NaN values appear
   df.sort_values('column', na_position='first')  # or 'last'
   ```

3. **Stable Sorting**:
   ```python
   # Maintain relative order of equal values
   df.sort_values(['A', 'B'], kind='stable')
   ```
{% endstep %}

{% step %}
### Ranking Best Practices
1. **Choose Appropriate Method**:
   ```python
   # For competition rankings (1224 ranking)
   df['Rank'] = df['Score'].rank(method='min')
   
   # For dense rankings (1223 ranking)
   df['Rank'] = df['Score'].rank(method='dense')
   
   # For unique rankings
   df['Rank'] = df['Score'].rank(method='first')
   ```

2. **Handle Percentile Rankings**:
   ```python
   # Calculate percentile ranks
   df['Percentile'] = df['Score'].rank(pct=True)
   ```
{% endstep %}
{% endstepper %}

## Common Pitfalls and Solutions

1. **Forgetting to Handle NaN Values**:
   ```python
   # Specify na_position explicitly
   df.sort_values('column', na_position='last')
   ```

2. **Incorrect Rank Method**:
   ```python
   # Different methods for different needs:
   # 'average': Default, assigns average of ranks for ties
   # 'min': Assigns minimum rank for ties
   # 'max': Assigns maximum rank for ties
   # 'first': Assigns ranks in order they appear
   # 'dense': Leaves no gaps in ranking
   ```

3. **Not Considering Performance**:
   ```python
   # More efficient for large datasets
   df.nlargest(10, 'column')  # Instead of sort_values().head(10)
   df.nsmallest(10, 'column')  # Instead of sort_values().tail(10)
   ```

Remember: Choose sorting and ranking methods based on your specific needs. Consider how you want to handle ties and missing values before applying these operations!
