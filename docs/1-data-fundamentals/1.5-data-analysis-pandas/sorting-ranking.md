# Sorting and Ranking in Pandas

**After this lesson:** you can explain Sorting and Ranking in Pandas and try the examples in your own notebook.

### Video

_Corey Schafer, Python pandas tutorial (part 7): sorting data_

## Overview

**Prerequisites:** [DataFrame](dataframe.md) indexing; basic notion of **rank** vs **sort**.

**Why this lesson:** Sorting puts rows in a readable order; **ranking** assigns positions for scoring and percentiles. You will use both for leaderboards, time-ordered panels, and preparing data for cumulative plots or window functions.

## Understanding Sorting

***

### What is Sorting?

Sorting in Pandas helps you organize your data in a specific order. Think of it like:

_Use `sort_values` to reorder rows for display or iteration. Use `rank` when you need a numeric position (e.g. leaderboard position, percentile) as a new column alongside the original data._

* Arranging books alphabetically on a shelf
* Organizing test scores from highest to lowest
* Arranging dates from oldest to newest
* Sorting transactions by amount
* Ranking players by score

Key benefits:

* Quick value lookup
* Pattern identification
* Data presentation
* Priority identification
* Trend analysis

Real-world applications:

* Financial portfolio analysis
* Sports rankings and statistics
* Sales performance reports
* Event scheduling and planning
* Customer segmentation

***

### Basic Sorting Example

we will look at sorting with practical examples:

**Sort Series by value or index; sort DataFrame by column**

* **Purpose:** Use `sort_values` on a Series and on a derived `Total` column; use `sort_index` for alphabetical names.
* **Walkthrough:** `ascending=False` puts largest totals first; `sort_index()` orders the **index** labels, not values.

```
Original scores:
Alice      85
Bob        92
Charlie    78
David      95
Eve        88
Name: Test Scores, dtype: int64

Top performers:
David      95
Bob        92
Eve        88
Alice      85
Charlie    78
Name: Test Scores, dtype: int64

Alphabetical order:
Alice      85
Bob        92
Charlie    78
David      95
Eve        88
Name: Test Scores, dtype: int64

Sales Data (sorted by total sales):
    Product  Price  Units       Date  Total
4    Laptop   1100      8 2023-01-05   8800
0    Laptop   1200      5 2023-01-01   6000
2  Keyboard    100     30 2023-01-03   3000
3   Monitor    300     10 2023-01-04   3000
1     Mouse     25     50 2023-01-02   1250
```

Series Sorting

Creates a named Series with student names as the index, then demonstrates `sort_values` for ranking by score and `sort_index` for alphabetical order.

DataFrame Sort

Builds a sales DataFrame, computes a Total column, then sorts descending by Total, showing how row order changes while all columns travel together.

```
Original scores:
Alice      85
Bob        92
Charlie    78
David      95
Eve        88
Name: Test Scores, dtype: int64

Top performers:
David      95
Bob        92
Eve        88
Alice      85
Charlie    78
Name: Test Scores, dtype: int64

Alphabetical order:
Alice      85
Bob        92
Charlie    78
David      95
Eve        88
Name: Test Scores, dtype: int64

Sales Data (sorted by total sales):
    Product  Price  Units       Date  Total
4    Laptop   1100      8 2023-01-05   8800
0    Laptop   1200      5 2023-01-01   6000
2  Keyboard    100     30 2023-01-03   3000
3   Monitor    300     10 2023-01-04   3000
1     Mouse     25     50 2023-01-02   1250
```

## Sorting DataFrames

***

### Sorting by a Single Column

Work with a student grades DataFrame:

**`sort_values` on one column**

* **Purpose:** Order rows by `Math` ascending or descending while keeping each student's row intact.
* **Walkthrough:** Default `ascending=True` gives low-to-high scores.

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

```
Original grades:
      Name  Math  Science  History
0    Alice    85       92       88
1      Bob    92       85       85
2  Charlie    78       96       92
3    David    95       88       85
4      Eve    88       90       95

Sorted by Math scores:
      Name  Math  Science  History
2  Charlie    78       96       92
0    Alice    85       92       88
4      Eve    88       90       95
1      Bob    92       85       85
3    David    95       88       85

Sorted by Math scores (highest to lowest):
      Name  Math  Science  History
3    David    95       88       85
1      Bob    92       85       85
4      Eve    88       90       95
0    Alice    85       92       88
2  Charlie    78       96       92
```

***

### Sorting by Multiple Columns

You can sort by multiple columns to break ties:

**Lexicographic sort with `ascending` list**

* **Purpose:** Break ties on `Science` using `Math` as secondary key; flip directions per column with `ascending=[False, True]`.
* **Walkthrough:** Column order in the list is **primary → secondary** sort.

```python
# Sort by Science first, then by Math
print("Sorted by Science, then Math:")
print(grades.sort_values(['Science', 'Math']))

# Sort Science descending and Math ascending
print("\nScience descending, Math ascending:")
print(grades.sort_values(['Science', 'Math'],
                        ascending=[False, True]))
```

```
Sorted by Science, then Math:
      Name  Math  Science  History
1      Bob    92       85       85
3    David    95       88       85
4      Eve    88       90       95
0    Alice    85       92       88
2  Charlie    78       96       92

Science descending, Math ascending:
      Name  Math  Science  History
2  Charlie    78       96       92
0    Alice    85       92       88
4      Eve    88       90       95
3    David    95       88       85
1      Bob    92       85       85
```

## Understanding Ranking

***

### What is Ranking?

Ranking assigns positions to your data based on their values. Think of it like:

* Ranking athletes in a competition
* Assigning class rank to students
* Determining the position of teams in a league

The difference from sorting is that ranking keeps your data in its original order but adds rank numbers.

***

### Basic Ranking Example

Look at different ways to rank data:

**Tie-breaking: average, first, min, max**

* **Purpose:** See how the same scores get different rank numbers depending on `method`-important for leaderboards and percentiles.
* **Walkthrough:** Tied `85` values get ranks 1.5 (average), 1 vs 2 (first), 1 (min), 2 (max).

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

```
Original scores:
0    85
1    92
2    85
3    95
4    88
dtype: int64

Default ranking:
0    1.5
1    4.0
2    1.5
3    5.0
4    3.0
dtype: float64

Rank with 'first' method:
0    1.0
1    4.0
2    2.0
3    5.0
4    3.0
dtype: float64

Rank with 'min' method:
0    1.0
1    4.0
2    1.0
3    5.0
4    3.0
dtype: float64

Rank with 'max' method:
0    2.0
1    4.0
2    2.0
3    5.0
4    3.0
dtype: float64
```

Notice how different methods handle the tied scores (85 appears twice).

## Real-World Examples

***

### Sales Performance Analysis

Analyze sales data:

**Global sort + `groupby` rank**

* **Purpose:** Sort for reporting, then assign **within-group** ranks (here by `Region`) so North and South are ranked separately.
* **Walkthrough:** `groupby('Region')['Sales'].rank(ascending=False)` broadcasts ranks back to original row order.

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

```
Sales data sorted by amount:
  Salesperson Region   Sales  Clients
3        Lisa  South  300000       60
4         Tom  North  250000       55
1       Sarah  South  200000       40
0        John  North  150000       50
2        Mike  North  150000       45

Sales with regional rankings:
  Salesperson Region   Sales  Clients  RegionalRank
0        John  North  150000       50           2.5
1       Sarah  South  200000       40           2.0
2        Mike  North  150000       45           2.5
3        Lisa  South  300000       60           1.0
4         Tom  North  250000       55           1.0
```

***

### Student Performance Analysis

Analyze student rankings across different subjects:

**Row-wise mean and per-subject ranks**

* **Purpose:** Combine `mean(axis=1)` for an overall score with per-column `rank` to show strengths in each subject.
* **Walkthrough:** `sort_values('OverallRank')` at the end orders by overall performance.

\`\`\` Student rankings: Name Math Science ... MathRank ScienceRank HistoryRank 4 Eve 88 90 ... 3.0 3.0 1.0 3 David 95 88 ... 1.0 4.0 4.5 2 Charlie 78 96 ... 5.0 1.0 2.0 0 Alice 85 92 ... 4.0 2.0 3.0 1 Bob 92 85 ... 2.0 5.0 4.5

\[5 rows x 9 columns]

```


</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Student DataFrame</span>
    </div>
    <div class="code-callout__body">
      <p>Creates a five-student table with three subject scores, enough to show ties and rank differences across subjects.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="9-11" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Overall Rank</span>
    </div>
    <div class="code-callout__body">
      <p>Computes a row-wise mean across the three subjects then ranks descending, highest average gets rank 1.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-18" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Per-Subject Ranks</span>
    </div>
    <div class="code-callout__body">
      <p>Loops over the three subjects to add an individual rank column for each, then sorts by overall rank so the top student appears first.</p>
    </div>
  </div>
</aside>
</div>

```

Student rankings: Name Math Science ... MathRank ScienceRank HistoryRank 4 Eve 88 90 ... 3.0 3.0 1.0 3 David 95 88 ... 1.0 4.0 4.5 2 Charlie 78 96 ... 5.0 1.0 2.0 0 Alice 85 92 ... 4.0 2.0 3.0 1 Bob 92 85 ... 2.0 5.0 4.5

\[5 rows x 9 columns]

````

## Best Practices and Tips

---

### Sorting Best Practices

1. **Preserve Original Data**:

   **Non-mutating vs in-place sort**

   - **Purpose:** Prefer `sort_values` returning a new frame unless you intentionally overwrite with `inplace=True`.

   ```python
   # Create sorted view without modifying original
   sorted_df = df.sort_values('column')

   # Or sort in-place if needed
   df.sort_values('column', inplace=True)
````

2.  **Handle Missing Values**:

    **Placement of NaNs when sorting**

    * **Purpose:** Decide whether missing keys sort to the top or bottom for dashboards.

    ```python
    # Control where NaN values appear
    df.sort_values('column', na_position='first')  # or 'last'
    ```
3.  **Stable Sorting**:

    **Stable sort for tied keys**

    * **Purpose:** When primary/secondary keys repeat, preserve original row order among ties, helps reproducible exports.

    ```python
    # Maintain relative order of equal values
    df.sort_values(['A', 'B'], kind='stable')
    ```

***

### Ranking Best Practices

1.  **Choose Appropriate Method**:

    ```python
    # For competition rankings (1224 ranking)
    df['Rank'] = df['Score'].rank(method='min')

    # For dense rankings (1223 ranking)
    df['Rank'] = df['Score'].rank(method='dense')

    # For unique rankings
    df['Rank'] = df['Score'].rank(method='first')
    ```
2.  **Handle Percentile Rankings**:

    ```python
    # Calculate percentile ranks
    df['Percentile'] = df['Score'].rank(pct=True)
    ```

## Common Pitfalls and Solutions

1.  **Forgetting to Handle NaN Values**:

    ```python
    # Specify na_position explicitly
    df.sort_values('column', na_position='last')
    ```
2.  **Incorrect Rank Method**:

    ```python
    # Different methods for different needs:
    # 'average': Default, assigns average of ranks for ties
    # 'min': Assigns minimum rank for ties
    # 'max': Assigns maximum rank for ties
    # 'first': Assigns ranks in order they appear
    # 'dense': Leaves no gaps in ranking
    ```
3.  **Not Considering Performance**:

    ```python
    # More efficient for large datasets
    df.nlargest(10, 'column')  # Instead of sort_values().head(10)
    df.nsmallest(10, 'column')  # Instead of sort_values().tail(10)
    ```

Remember: Choose sorting and ranking methods based on your specific needs. Consider how you want to handle ties and missing values before applying these operations!

## Next steps

Continue to [Arithmetic and alignment](arithmetic-alignment.md) to finish core pandas patterns in this submodule.
