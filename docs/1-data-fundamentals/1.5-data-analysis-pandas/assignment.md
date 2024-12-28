# Assignment: Data Analysis with Pandas

## Setup

First, create the following mock e-commerce dataset using this code:

```python
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(2024)

# Create a mock dataset
data = {
    'order_id': range(1, 11),
    'customer_id': np.random.randint(1000, 1020, size=10),
    'product_id': np.random.randint(100, 110, size=10),
    'quantity': np.random.randint(1, 5, size=10),
    'price': np.random.uniform(10.0, 100.0, size=10),
    'order_date': pd.date_range(start='2021-01-01', periods=10, freq='D')
}

df = pd.DataFrame(data)
print(df)
```

## Tasks

### 1. Basic Data Exploration

1. Display the data types of each column in the DataFrame
2. Calculate basic statistics (mean, min, max) for the 'quantity' column
3. Check if there are any missing values in the DataFrame

### 2. Data Manipulation & Arithmetic

1. Create a new column 'total_amount' by multiplying 'quantity' and 'price'
2. Calculate the daily revenue (sum of total_amount) and store it in a new Series
3. Add 5% tax to all prices and store in a new column 'price_with_tax'
4. Find orders where the quantity is above the mean quantity

### 3. Sorting & Ranking

1. Sort the DataFrame by total_amount in descending order
2. Rank the orders based on their price (highest price = rank 1)
3. Find the top 3 orders by total_amount
4. Sort the orders by date and quantity

### 4. Function Application

1. Create a function that categorizes total_amount into 'High' (>$200), 'Medium' ($100-$200), and 'Low' (<$100)
2. Apply this function to create a new column 'order_category'
3. Format the price and total_amount columns to display as currency with 2 decimal places
4. Calculate the cumulative sum of total_amount ordered by date

### 5. Index Operations

1. Set the order_date as the index of the DataFrame
2. Select all orders from the first 5 days
3. Reset the index back to default numeric indices
4. Create a new copy of the DataFrame with order_id as the index

## Instructions

1. Complete each task in order
2. Document your code with comments
3. Use appropriate pandas methods and functions
4. Format your output for readability

## Deliverable

Submit a Jupyter notebook containing:

- The setup code
- All task solutions with explanations
- A brief summary of insights found in the data

Your notebook should be well-organized with markdown cells or comments in code explaining your approach for each task.
