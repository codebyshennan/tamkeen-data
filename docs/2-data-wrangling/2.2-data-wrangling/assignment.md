# Data Wrangling Assessment 🔍

## Part 1: Theoretical Understanding (30 points)

### Multiple Choice Questions (15 points)

1. What is data cleaning? (3 points)
   a. Collecting data from various sources
   b. Analyzing data patterns and trends
   c. Fixing or removing incorrect, corrupted, or irrelevant data
   d. Visualizing data relationships
   
   **Explanation**: Data cleaning is the process of identifying and correcting (or removing) errors in datasets. This includes handling missing values, removing duplicates, fixing inconsistencies, and ensuring data quality.

2. Which of these is NOT a common data quality issue? (3 points)
   a. Missing values in customer records
   b. Duplicate transaction entries
   c. Inconsistent date formats
   d. Beautiful data visualization
   
   **Explanation**: While missing values, duplicates, and inconsistent formats are common data quality issues that need to be addressed during data wrangling, data visualization is a separate aspect of data analysis, not a quality issue.

3. What is the best approach for handling missing values? (3 points)
   a. Always delete rows with missing values
   b. Always replace with zero
   c. Choose method based on context and data patterns
   d. Ignore them completely
   
   **Explanation**: The appropriate method for handling missing values depends on various factors:
   - The amount of missing data
   - The pattern of missingness (MCAR, MAR, MNAR)
   - The importance of the missing information
   - The potential impact on analysis

4. What is data transformation? (3 points)
   a. Converting data from one format or structure to another
   b. Deleting incorrect data entries
   c. Collecting new data samples
   d. Analyzing data patterns
   
   **Explanation**: Data transformation involves converting data from its raw form into a format more suitable for analysis. This includes:
   - Scaling numerical features
   - Encoding categorical variables
   - Feature engineering
   - Format standardization

5. Which scaling method is most robust to outliers? (3 points)
   a. Standard scaling (z-score)
   b. Robust scaling (using quartiles)
   c. Min-max scaling
   d. Log transformation
   
   **Explanation**: Robust scaling uses statistics that are not influenced by outliers:
   - Uses median instead of mean
   - Uses interquartile range instead of standard deviation
   - Maintains relative relationships while reducing outlier impact

### Short Answer Questions (15 points)

6. Explain three different strategies for handling outliers and when each might be appropriate. (5 points)

   Example answer:
   ```
   1. Removal: When outliers are clearly errors
      - Example: Age = 999 years in customer data
   
   2. Capping: When outliers are extreme but possible
      - Example: Setting very high sales amounts to 95th percentile
   
   3. Transformation: When data is skewed but outliers are valid
      - Example: Log transform of highly skewed income data
   ```

7. Describe the process of feature engineering and provide two examples. (5 points)

   Example answer:
   ```
   Feature engineering is creating new features from existing data to improve model performance:
   
   1. Time-based features:
      - Converting timestamp to hour/day/month
      - Creating "is_weekend" flag
   
   2. Interaction features:
      - Price per unit from total_price/quantity
      - Customer lifetime value from purchase history
   ```

8. What are the key considerations when choosing a data imputation strategy? (5 points)

   Example answer:
   ```
   Key considerations include:
   1. Missing data mechanism (MCAR, MAR, MNAR)
   2. Amount of missing data (percentage)
   3. Relationships between variables
   4. Domain knowledge and constraints
   5. Impact on downstream analysis
   ```

## Part 2: Practical Application (70 points)

### Case Study: E-commerce Data Cleaning

You are provided with a messy e-commerce dataset containing customer transactions. The data has various quality issues that need to be addressed.

```python
# Sample data structure
transactions_df = pd.DataFrame({
    'customer_id': [1, 2, np.nan, 4, 5],
    'purchase_date': ['2023-01-01', '2023-13-01', '2023-01-03', '2023-01-04', '2023-01-05'],
    'amount': [100, -50, 1000000, 200, 150],
    'product_category': ['Electronics', 'electronics', np.nan, 'Clothing', 'Electronics']
})
```

Tasks:

1. Data Quality Assessment (20 points)
   - Identify all data quality issues
   - Document your findings
   - Propose cleaning strategies

2. Data Cleaning Implementation (30 points)
   - Handle missing values
   - Fix invalid dates
   - Address negative amounts
   - Standardize categories
   - Remove or handle outliers

3. Data Validation (20 points)
   - Implement checks to verify cleaning
   - Document any assumptions made
   - Provide summary statistics before and after

Submit your solution as a Jupyter notebook with:
- Clear documentation
- Code implementation
- Results visualization
- Quality validation

## Evaluation Criteria

- Code quality and documentation (20%)
- Proper handling of each data issue (40%)
- Validation and testing (20%)
- Explanation of decisions (20%)

Good luck! Remember to think critically about each decision and document your reasoning! 🚀
