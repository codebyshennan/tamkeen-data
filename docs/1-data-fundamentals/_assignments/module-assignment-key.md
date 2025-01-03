# Module 1: Data Fundamentals - Answer Key

## Part 1: Introduction to Data Analytics (20 points)

1. What is the primary purpose of data collection?
   - a. To analyze data
   - b. To gather raw material for analysis ✓
   - c. To ensure data accuracy
   - d. To visualize data

2. Which of the following is an example of first-party data?
   - a. Social media insights
   - b. Market research reports
   - c. Customer interactions ✓
   - d. Data aggregators

3. What does GDPR stand for?
   - a. General Data Protection Regulation ✓
   - b. General Data Privacy Regulation
   - c. General Data Protection Regulation
   - d. Global Data Protection Regulation

4. Which method is used for gathering qualitative data?
   - a. Surveys
   - b. Data logging
   - c. Interviews ✓
   - d. Web scraping

5. What is the main focus of predictive analytics?
   - a. Understanding past events
   - b. Analyzing current data
   - c. Forecasting future outcomes ✓
   - d. Summarizing data

6. What does PII stand for?
   - a. Personal Information Identifier
   - b. Protected Information Index
   - c. Personally Identifiable Information ✓
   - d. Private Information Indicator

7. Which of the following is a key principle of data security?
   - a. Data Minimization
   - b. Purpose Limitation
   - c. Confidentiality ✓
   - d. Data Portability

8. What is the purpose of data encryption?
   - a. To analyze data
   - b. To visualize data
   - c. To prevent unauthorized access ✓
   - d. To store data

9. Which lifecycle stage involves removing records with missing data?
   - a. Data Cleaning ✓
   - b. Data Exploration
   - c. Data Collection
   - d. Data Analysis

10. What is the main goal of data cleaning?
    - a. To visualize data
    - b. To collect data
    - c. To remove inaccuracies ✓
    - d. To analyze data

11. What is the role of feature engineering in data science?
    - a. To collect data
    - b. To clean data
    - c. To create new features ✓
    - d. To visualize data

12. Which of the following is a type of secondary data analysis?
    - a. Surveys
    - b. Interviews
    - c. Analyzing existing datasets ✓
    - d. Observations

13. What is the primary focus of the Data Science Lifecycle?
    - a. Data Collection
    - b. Data Cleaning
    - c. Developing and delivering data science projects ✓
    - d. Data Visualization

14. What does the term "data minimization" refer to?
    - a. Collecting as much data as possible
    - b. Storing data indefinitely
    - c. Collecting only necessary data ✓
    - d. Analyzing data

15. Which of the following is a threat to data security?
    - a. Data encryption
    - b. Data backup
    - c. Malware ✓
    - d. Access control

16. What is the purpose of a feedback loop in data analytics?
    - a. To collect data
    - b. To clean data
    - c. To refine the analysis process ✓
    - d. To visualize data

17. Which of the following is an example of third-party data?
    - a. CRM data
    - b. Customer interactions
    - c. Data from data aggregators ✓
    - d. Surveys

18. What is the main benefit of using web scraping?
    - a. To analyze data
    - b. To conduct interviews
    - c. To collect large datasets quickly ✓
    - d. To visualize data

19. What is the significance of the right to erasure under data privacy laws?
    - a. Individuals can access their data
    - b. Individuals can correct their data
    - c. Individuals can request deletion of their data ✓
    - d. Individuals can transfer their data

20. Which of the following best describes the role of data visualization?
    - a. To collect data
    - b. To clean data
    - c. To present data and analysis results ✓
    - d. To analyze data

## Part 2: Python Programming (20 points)

```python
# Task 1: Data Types and Variables
num = 42                                  # Integer
pi = 3.14159                             # Float
message = "Hello Python!"                 # String
is_active = True                         # Boolean
fruits = ["apple", "banana", "orange"]    # List
coordinates = (10, 20)                    # Tuple
person = {                               # Dictionary
    "name": "John",
    "age": 30,
    "city": "New York"
}

# Task 2: Functions and Classes
def count_and_return_vowels(text):
    """
    Count and return vowels in the given text
    """
    vowels = 'aeiouAEIOU'
    found_vowels = [char for char in text if char in vowels]
    return len(found_vowels), found_vowels

def sum_of_even_numbers(limit):
    """
    Calculate sum of even numbers up to limit
    """
    total = 0
    num = 0
    while num <= limit:
        if num % 2 == 0:
            total += num
        num += 1
    return total

class BankAccount:
    """
    Bank account class implementation
    """
    def __init__(self, initial_balance):
        self.balance = initial_balance
        
    def deposit(self, amount):
        self.balance += amount
        
    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
        else:
            print("Insufficient funds")
            
    def get_balance(self):
        return self.balance

# Test implementations
print("Testing count_and_return_vowels:")
print(count_and_return_vowels("Hello World"))  # Output: (3, ['e', 'o', 'o'])
print(count_and_return_vowels("Python"))       # Output: (1, ['o'])

print("\nTesting sum_of_even_numbers:")
print(sum_of_even_numbers(10))  # Output: 30 (2+4+6+8+10)
print(sum_of_even_numbers(5))   # Output: 6 (2+4)

print("\nTesting BankAccount:")
account = BankAccount(100)
print(account.get_balance())    # Output: 100
account.deposit(50)
print(account.get_balance())    # Output: 150
account.withdraw(30)
print(account.get_balance())    # Output: 120
account.withdraw(200)           # Output: "Insufficient funds"
print(account.get_balance())    # Output: 120
```

## Part 3: Statistics (20 points)

1. In a dataset with values [2, 4, 4, 6, 8, 8, 8, 10], what is the mode?
   - a. 4
   - b. 6
   - c. 8 ✓
   - d. 10

2. If a dataset has a mean greater than its median, the distribution is likely:
   - a. Symmetrical
   - b. Negatively skewed
   - c. Positively skewed ✓
   - d. Normal

3. The correlation coefficient ranges from:
   - a. -1 to 1 ✓
   - b. 0 to 1
   - c. -1 to 0
   - d. -2 to 2

4. Which of the following is not a measure of central tendency?
   - a. Mean
   - b. Median
   - c. Mode
   - d. Range ✓

5. In a normal distribution, approximately what percentage of data falls within one standard deviation of the mean?
   - a. 50%
   - b. 68% ✓
   - c. 95%
   - d. 99%

6. The interquartile range (IQR) is calculated as:
   - a. Q1 - Q3
   - b. Q3 - Q1 ✓
   - c. Q2 - Q1
   - d. Q3 - Q2

7. Which probability distribution is used to model the number of successes in a fixed number of trials?
   - a. Normal distribution
   - b. Binomial distribution ✓
   - c. Poisson distribution
   - d. Uniform distribution

8. What does the Central Limit Theorem state about sample means?
   - a. They are always normally distributed
   - b. They approach normal distribution as sample size increases ✓
   - c. They are always equal to the population mean
   - d. They are always skewed

9. In a Poisson distribution, which of the following is true?
   - a. The mean equals the variance ✓
   - b. The mean is greater than the variance
   - c. The mean is less than the variance
   - d. The mean and variance are unrelated

10. The probability density function is used for:
    - a. Discrete distributions only
    - b. Continuous distributions only ✓
    - c. Both discrete and continuous distributions
    - d. Neither discrete nor continuous distributions

11. In a normal distribution, the mean, median, and mode are:
    - a. Always different
    - b. All equal ✓
    - c. Mean equals median only
    - d. Mode equals median only

12. The Z-score represents:
    - a. The probability of an event
    - b. The number of standard deviations from the mean ✓
    - c. The mean of a distribution
    - d. The variance of a distribution

13. Correlation measures:
    - a. The strength and direction of a linear relationship ✓
    - b. Causation between variables
    - c. The slope of a line
    - d. The variance between variables

14. A covariance value of zero indicates:
    - a. Perfect positive correlation
    - b. Perfect negative correlation
    - c. No linear relationship ✓
    - d. Strong relationship

15. In a scatter plot, a positive correlation shows:
    - a. Points moving downward from left to right
    - b. Points moving upward from left to right ✓
    - c. Points in a random pattern
    - d. Points in a circular pattern

16. Which measure of central tendency is most affected by outliers?
    - a. Mean ✓
    - b. Median
    - c. Mode
    - d. Range

17. The Poisson distribution is used to model:
    - a. Number of successes in fixed trials
    - b. Number of events in a fixed interval ✓
    - c. Continuous data only
    - d. Normally distributed data

18. In a negatively skewed distribution:
    - a. The mean is greater than the median
    - b. The mean equals the median
    - c. The mean is less than the median ✓
    - d. The mode is always zero

19. For a binomial distribution with n trials and probability p, the mean is:
    - a. n × p ✓
    - b. n + p
    - c. n ÷ p
    - d. n - p

20. The standard deviation is calculated by:
    - a. Subtracting each value from the mean and dividing by n
    - b. Finding the square root of the variance ✓
    - c. Taking the absolute difference between maximum and minimum values
    - d. Multiplying the mean by the sample size

## Part 4: NumPy Operations (20 points)

```python
import numpy as np

# Setup
scores = np.array([
    [85, 92, 78],
    [90, 88, 95],
    [75, 70, 85],
    [88, 95, 92],
    [65, 72, 68],
    [95, 88, 85],
    [78, 85, 82],
    [92, 89, 90]
])

names = np.array(['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry'])

matrix_A = np.random.randint(1, 10, size=(4, 4))
matrix_B = np.random.randint(1, 10, size=(4, 4))

# 1. Array Operations and Indexing
# Calculate average scores
avg_scores = scores.mean(axis=1)
print("Average scores per student:")
for name, avg in zip(names, avg_scores):
    print(f"{name}: {avg:.2f}")

# Find highest scores
max_scores = scores.max(axis=0)
subjects = ['Math', 'Science', 'English']
print("\nHighest scores per subject:")
for subject, max_score in zip(subjects, max_scores):
    print(f"{subject}: {max_score}")

# Students scoring above 90
high_scorers = names[np.any(scores > 90, axis=1)]
print("\nStudents with scores above 90:", high_scorers)

# Students passing all subjects
passing_mask = np.all(scores >= 70, axis=1)
passing_students = names[passing_mask]
print("\nStudents passing all subjects:", passing_students)

# 2. Array Manipulation
# Reshape scores
reshaped_scores = scores.reshape(12, 2)
print("\nReshaped scores:\n", reshaped_scores)

# Standardize scores
standardized = (scores - scores.mean(axis=0)) / scores.std(axis=0)
print("\nStandardized scores:\n", standardized)

# Sort by average score
sort_indices = avg_scores.argsort()[::-1]
sorted_names = names[sort_indices]
sorted_scores = scores[sort_indices]
print("\nStudents sorted by average score:")
for name, score in zip(sorted_names, sorted_scores):
    print(f"{name}: {score.mean():.2f}")

# Statistics per subject
print("\nSubject statistics:")
for i, subject in enumerate(subjects):
    print(f"{subject}:")
    print(f"Min: {scores[:,i].min()}")
    print(f"Max: {scores[:,i].max()}")
    print(f"Mean: {scores[:,i].mean():.2f}")

# 3. Linear Algebra
# Matrix multiplication
matrix_product = np.matmul(matrix_A, matrix_B)
print("\nMatrix product:\n", matrix_product)

# Determinant
det_A = np.linalg.det(matrix_A)
print("\nDeterminant of matrix_A:", det_A)

# Matrix inverse
try:
    inv_A = np.linalg.inv(matrix_A)
    print("\nInverse of matrix_A:\n", inv_A)
except np.linalg.LinAlgError:
    print("\nMatrix_A is not invertible")

# Eigenvalues
eigenvals = np.linalg.eigvals(matrix_A)
print("\nEigenvalues of matrix_A:", eigenvals)

# 4. Bonus Challenge
def student_analysis(student_name):
    # Find student index
    student_idx = np.where(names == student_name)[0][0]
    
    # Get scores
    student_scores = scores[student_idx]
    
    # Calculate rankings
    rankings = []
    for i in range(3):
        ranking = len(scores[:,i]) - np.searchsorted(np.sort(scores[:,i]), student_scores[i])
        rankings.append(ranking)
    
    # Check if in top 3
    avg_scores = scores.mean(axis=1)
    top_3 = np.argsort(avg_scores)[-3:]
    is_top_3 = student_idx in top_3
    
    return {
        'scores': student_scores,
        'rankings': rankings,
        'is_top_3': is_top_3
    }

# Example usage
result = student_analysis('Alice')
print("\nAnalysis for Alice:", result)
```

## Part 5: Pandas Data Analysis (20 points)

```python
import pandas as pd
import numpy as np

# Setup
np.random.seed(2024)
data = {
    'order_id': range(1, 11),
    'customer_id': np.random.randint(1000, 1020, size=10),
    'product_id': np.random.randint(100, 110, size=10),
    'quantity': np.random.randint(1, 5, size=10),
    'price': np.random.uniform(10.0, 100.0, size=10),
    'order_date': pd.date_range(start='2021-01-01', periods=10, freq='D')
}

df = pd.DataFrame(data)

# 1. Basic Data Exploration
print("Data types:")
print(df.dtypes)

print("\nQuantity statistics:")
print(df['quantity'].describe())

print("\nMissing values:")
print(df.isnull().sum())

# 2. Data Manipulation & Arithmetic
df['total_amount'] = df['quantity'] * df['price']

daily_revenue = df.groupby('order_date')['total_amount'].sum()
print("\nDaily revenue:")
print(daily_revenue)

df['price_with_tax'] = df['price'] * 1.05

above_mean_qty = df[df['quantity'] > df['quantity'].mean()]
print("\nOrders with above-mean quantity:")
print(above_mean_qty)

# 3. Sorting & Ranking
df_sorted = df.sort_values('total_amount', ascending=False)
print("\nSorted by total amount:")
print(df_sorted)

df['price_rank'] = df['price'].rank(ascending=False)
print("\nPrice rankings:")
print(df[['order_id', 'price', 'price_rank']])

top_3 = df.nlargest(3, 'total_amount')
print("\nTop 3 orders:")
print(top_3)

df_sorted_date_qty = df.sort_values(['order_date', 'quantity'])
print("\nSorted by date and quantity:")
print(df_sorted_date_qty)

# 4. Function Application
def categorize_amount(amount):
    if amount > 200:
        return 'High'
    elif amount >= 100:
        return 'Medium'
    return 'Low'

df['order_category'] = df['total_amount'].apply(categorize_amount)

df['price_formatted'] = df['price'].map('${:,.2f}'.format)
df['total_amount_formatted'] = df['total_amount'].map('${:,.2f}'.format)

df['cumulative_amount'] = df.sort_values('order_date')['total_amount'].cumsum()
print("\nDataFrame with new columns:")
print(df)

# 5. Index Operations
df_dated = df.set_index('order_date')
print("\nDataFrame with date index:")
print(df_dated)

first_5_days = df_dated.iloc[:5]
print("\nFirst 5 days:")
print(first_5_days)

df_reset = df_dated.reset_index()
print("\nReset index:")
print(df_reset)

df_ordered = df.set_index('order_id')
print("\nDataFrame with order_id index:")
print(df_ordered)
```

## Grading Rubric

Each part is worth 20 points:

### Part 1: Data Analytics Quiz
- 1 point per correct answer (20 questions)

### Part 2: Python Programming
- Correct implementation of data types (5 points)
- Correct implementation of functions (7 points)
- Correct implementation of class (8 points)

### Part 3: Statistics Quiz
- 1 point per correct answer (20 questions)

### Part 4: NumPy Operations
- Array Operations and Indexing (5 points)
- Array Manipulation (5 points)
- Linear Algebra (5 points)
- Bonus Challenge (5 points)

### Part 5: Pandas Analysis
- Basic Data Exploration (4 points)
- Data Manipulation & Arithmetic (4 points)
- Sorting & Ranking (4 points)
- Function Application (4 points)
- Index Operations (4 points)

Total possible points: 100
