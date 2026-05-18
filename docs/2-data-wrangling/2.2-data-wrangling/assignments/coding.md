# Assignment: Data Wrangling with Pandas

## Setup

Run the following code to create your dataset:

```python
import pandas as pd
import numpy as np

# Create a messy dataset
data = pd.DataFrame({
    'name': ['John Smith', 'Jane Doe', 'Bob Wilson', 'MARY JONES', 'john smith', np.nan, 'Jane doe'],
    'email': ['john@gmail.com', 'jane@yahoo.com', np.nan, 'mary@gmail.com', 'john2@gmail.com', 'unknown@test.com', 'jane@yahoo.com'],
    'age': [25, 30, -999, 35, 25, 40, 30],
    'category': ['A', 'B', 'A', 'C', 'A', 'B', 'B'],
    'score': [85.5, 90.0, 77.5, 995.0, 85.5, 88.0, 90.0]
})
```

## Tasks

1. Data Cleaning:

   - Remove duplicate rows based on the 'name' column (case-insensitive)
   - Replace the age value -999 with NaN
   - Remove any rows with missing names

2. String Manipulation:

   - Convert all names to title case
   - Create a new column 'domain' that extracts the domain part from email addresses (everything after @)
   - Create a boolean column 'is_gmail' that is True if the email is from gmail.com

3. Categorical Data:

   - Convert the 'category' column to categorical type
   - Create dummy variables for the category column
   - Display the frequency count of categories

4. Handling Outliers:

   - Find any scores that are more than 2 standard deviations from the mean
   - Replace these outlier scores with the mean score
   - Calculate and display descriptive statistics for the cleaned score column

5. Data Transformation:
   - Create a dictionary that maps categories to numeric values (A=1, B=2, C=3)
   - Add a new column 'category_num' using this mapping
   - Sort the DataFrame by category_num and score

## Deliverable

Submit your solution as a Python script with:

1. All code clearly commented with
   1. Brief explanations of your approach for complex operations
   2. Any assumptions or additional features you implemented
2. Results for each task printed with appropriate labels
