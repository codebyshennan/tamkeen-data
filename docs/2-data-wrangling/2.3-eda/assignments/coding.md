# Exploratory Data Analysis Assignment

In this assignment, you'll work with a dataset containing information about flights. You'll apply various data analysis techniques covered in the lessons.

## Setup

First, run this code to generate the sample dataset:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate dates for 2023
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

# Create airlines and destinations
airlines = ['SIA', 'Emirates', 'Cathay', 'AirAsia', 'Qatar']
destinations = ['Tokyo', 'London', 'Dubai', 'Bangkok', 'Sydney']

# Generate random data
n_flights = 1000
data = {
    'date': np.random.choice(dates, n_flights),
    'airline': np.random.choice(airlines, n_flights),
    'destination': np.random.choice(destinations, n_flights),
    'departure_delay': np.random.normal(15, 30, n_flights),  # mean 15 min delay
    'arrival_delay': np.random.normal(20, 35, n_flights),    # mean 20 min delay
    'passengers': np.random.randint(100, 400, n_flights),
    'ticket_price': np.random.uniform(200, 1500, n_flights)
}

# Create DataFrame
flights = pd.DataFrame(data)

# Sort by date
flights = flights.sort_values('date')
```

## Tasks

1. **DateTime Analysis**

   - Convert the 'date' column to datetime if it isn't already
   - Calculate the average delays by month
   - Create a 7-day rolling average of passenger numbers
   - Find the busiest day of the week for each airline

2. **Correlation Analysis**

   - Calculate the correlation between departure_delay and arrival_delay
   - Find the correlation between passenger numbers and ticket prices
   - Identify which airline shows the strongest correlation between delays and ticket prices

3. **Data Reshaping**

   - Create a pivot table showing average ticket prices by airline and destination
   - Reshape the data to show daily passenger totals for each airline (wide format)
   - Convert the wide format back to long format

4. **Hierarchical Indexing**

   - Create a MultiIndex DataFrame grouping by airline and destination
   - Calculate average delays and prices for each group
   - Demonstrate at least two different ways to select data using the hierarchical index

5. **Data Combination**

   - Split the dataset into two parts (first 6 months and last 6 months)
   - Practice merging them back together using different join types
   - Create a new DataFrame with monthly summaries and merge it with the original data

6. **Advanced Aggregation**
   - Create a cross-tabulation of airlines and destinations showing flight counts
   - Calculate the percentage of delayed flights (>15 min) by airline
   - Use groupby and aggregate to calculate multiple statistics (min, max, mean, count) for numeric columns

## Deliverable

Submit your solution as a Python script with:

1. All code clearly commented with
   1. Brief explanations of your approach for complex operations
   2. Any assumptions or additional features you implemented
2. Results for each task printed with appropriate labels
