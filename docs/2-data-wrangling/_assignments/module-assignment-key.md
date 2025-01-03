# Module 2: Data Wrangling Assignment

This comprehensive assignment covers key concepts from SQL, Data Wrangling, Exploratory Data Analysis (EDA), and Data Engineering.

## Part 1: SQL Fundamentals

1. Which SQL command is used to retrieve data from a database?
   _a. SELECT_
   b. UPDATE
   c. INSERT
   d. DELETE

**Explanation**: The SELECT statement is the fundamental command for retrieving data from a database. It allows you to specify exactly which columns you want to retrieve and from which tables. While UPDATE modifies existing records, INSERT adds new records, and DELETE removes records, SELECT is specifically designed for data retrieval operations.

*For more information, see: [Basic SQL Operations](2.1-sql/basic-operations.md)*

2. What is the purpose of the WHERE clause?
   a. To sort data
   b. To group data
   _c. To filter data_
   d. To join tables

**Explanation**: The WHERE clause is used to filter records based on specified conditions. It acts like a filter that only allows rows meeting certain criteria to be included in the result set. For example, `WHERE age > 18` would only return records where the age column contains values greater than 18. This is different from sorting (ORDER BY), grouping (GROUP BY), or joining tables (JOIN).

3. Which JOIN type returns all records from both tables?
   a. INNER JOIN
   _b. FULL OUTER JOIN_
   c. LEFT JOIN
   d. RIGHT JOIN

**Explanation**: A FULL OUTER JOIN returns all records from both tables, matching records where possible and filling with NULL values where there is no match. This is different from:
- INNER JOIN: Only returns matching records
- LEFT JOIN: Returns all records from the left table and matching from right
- RIGHT JOIN: Returns all records from the right table and matching from left

4. What is the purpose of GROUP BY?
   a. To sort records
   _b. To group rows with similar values_
   c. To filter records
   d. To join tables

**Explanation**: `GROUP BY` is used to group rows that have the same values in specified columns into summary rows. It's typically used with aggregate functions (like `COUNT`, SUM, AVG) to perform calculations on each group rather than the entire table. For example, you might use GROUP BY to find the average salary per department or the total sales per region.

5. Which aggregate function returns the number of rows?
   a. SUM
   b. AVG
   _c. COUNT_
   d. MAX

**Explanation**: COUNT is an aggregate function that returns the number of rows that match a specified criteria. While:
- SUM adds up numeric values
- AVG calculates the mean of numeric values
- MAX finds the highest value
COUNT specifically deals with counting rows, making it essential for tasks like finding the number of customers, products, or transactions.

## Part 2: Data Wrangling

6. What is data cleaning?
   a. Collecting data
   b. Analyzing data
   _c. Fixing or removing incorrect data_
   d. Visualizing data

**Explanation**: Data cleaning is the crucial first step in the data preparation process. It involves:
- Identifying and correcting errors in datasets
- Handling missing or incomplete data
- Removing duplicate records
- Standardizing formats and units
- Dealing with outliers
This process ensures data quality and reliability for subsequent analysis. Poor data quality can lead to incorrect conclusions and decisions.

7. Which of these is NOT a common data quality issue?
   a. Missing values
   b. Duplicate records
   c. Inconsistent formats
   _d. Beautiful visualization_

**Explanation**: Beautiful visualization is not a data quality issue - it's related to data presentation. Common data quality issues include:
- Missing values: Gaps in the dataset
- Duplicate records: Repeated entries that can skew analysis
- Inconsistent formats: Like different date formats or units
These issues affect data integrity and must be addressed before analysis.

8. What is the best way to handle missing values?
   a. Always delete rows with missing values
   b. Always replace with zero
   _c. Choose method based on context and data_
   d. Ignore them

**Explanation**: The appropriate method for handling missing values depends on several factors:
- The type of data
- The amount of missing data
- The pattern of missingness
- The impact on analysis
Methods include:
- Deletion (when data is missing completely at random)
- Mean/median imputation (for numeric data)
- Mode imputation (for categorical data)
- Advanced methods like regression or multiple imputation

9. What is data transformation?
   _a. Converting data from one format to another_
   b. Deleting incorrect data
   c. Collecting new data
   d. Analyzing data patterns

**Explanation**: Data transformation involves changing the form or structure of data while preserving its meaning. Common transformations include:
- Normalization/Standardization
- Log transformations
- One-hot encoding
- Binning/Discretization
These transformations prepare data for analysis or modeling by making it more suitable for specific techniques.

10. Which scaling method preserves zero values and is robust to outliers?
    a. Standard scaling
    _b. Robust scaling_
    c. Min-max scaling
    d. Log transformation

**Explanation**: Robust scaling (also known as robust standardization) uses statistics that are robust to outliers:
- Uses median instead of mean
- Uses interquartile range (IQR) instead of standard deviation
- Preserves zero values
- Less influenced by extreme values
This makes it particularly useful for datasets with outliers or skewed distributions.

## Part 3: Exploratory Data Analysis (EDA)

11. What is the main purpose of EDA?
    _a. To understand patterns and relationships in data_
    b. To clean data
    c. To collect data
    d. To make predictions

**Explanation**: Exploratory Data Analysis (EDA) is a critical first step in data analysis that:
- Helps identify patterns, trends, and relationships in data
- Reveals potential anomalies or outliers
- Suggests hypotheses for further investigation
- Informs the choice of statistical methods
- Provides insights that guide more complex analyses
This systematic exploration helps analysts understand the fundamental characteristics of their dataset before proceeding with more advanced analyses.

12. Which plot is best for showing the distribution of a continuous variable?
    a. Bar plot
    b. Scatter plot
    _c. Histogram_
    d. Pie chart

**Explanation**: A histogram is ideal for continuous variables because it:
- Shows the frequency distribution of values
- Reveals the shape of the distribution (normal, skewed, bimodal, etc.)
- Helps identify outliers and gaps in the data
- Makes it easy to see central tendency and spread
Other plots serve different purposes:
- Bar plots are for categorical data
- Scatter plots show relationships between two variables
- Pie charts show proportions of a whole

13. What does a box plot show?
    a. Only the median
    _b. Five-number summary and outliers_
    c. Just the outliers
    d. Mean and standard deviation

**Explanation**: A box plot (or box-and-whisker plot) provides a comprehensive view of the data's distribution by showing:
- Median (middle line)
- First quartile (Q1, bottom of box)
- Third quartile (Q3, top of box)
- Minimum and maximum within 1.5 × IQR (whiskers)
- Individual points for outliers
This makes it excellent for comparing distributions and identifying outliers.

14. Which correlation coefficient ranges from -1 to 1?
    _a. Pearson_
    b. Chi-square
    c. F-statistic
    d. T-statistic

**Explanation**: The Pearson correlation coefficient:
- Ranges from -1 (perfect negative correlation) to +1 (perfect positive correlation)
- 0 indicates no linear correlation
- Measures the strength and direction of linear relationships
- Is widely used in statistical analysis
Other statistics mentioned serve different purposes:
- Chi-square tests categorical variable independence
- F-statistic compares variances
- T-statistic compares means

15. What type of plot is best for showing relationships between two continuous variables?
    _a. Scatter plot_
    b. Bar plot
    c. Line plot
    d. Pie chart

**Explanation**: A scatter plot is ideal for showing relationships between two continuous variables because it:
- Shows the pattern of relationship (linear, curved, none)
- Reveals the strength of the relationship
- Helps identify outliers
- Can show clustering or grouping
- Can be enhanced with trend lines or confidence intervals
Other plots are better suited for different purposes:
- Bar plots for categorical data
- Line plots for time series
- Pie charts for proportions

## Part 4: Data Engineering

16. What is ETL?
    _a. Extract, Transform, Load_
    b. Export, Transfer, Link
    c. Evaluate, Test, Launch
    d. Extract, Test, Load

**Explanation**: ETL (Extract, Transform, Load) is a fundamental data engineering process that:
- Extract: Retrieves data from various source systems
- Transform: Converts data into a suitable format and structure
- Load: Stores the processed data in a target system
This process is crucial for:
- Data integration
- Data warehouse population
- Data quality assurance
- Business intelligence support

17. Which storage format is best for structured data?
    a. Text files
    _b. Relational databases_
    c. PDF files
    d. Image files

**Explanation**: Relational databases are optimal for structured data because they:
- Enforce data integrity through schemas
- Support complex queries through SQL
- Provide ACID compliance (Atomicity, Consistency, Isolation, Durability)
- Enable efficient indexing and retrieval
- Support relationships between different data entities
Other formats serve different purposes:
- Text files: Simple data storage
- PDF files: Document storage
- Image files: Visual data storage

18. What is data integration?
    _a. Combining data from different sources_
    b. Cleaning data
    c. Analyzing data
    d. Visualizing data

**Explanation**: Data integration is the process of:
- Combining data from multiple sources
- Ensuring consistency across integrated data
- Resolving conflicts and duplications
- Creating a unified view of the data
- Maintaining data lineage
This is essential for:
- Creating comprehensive datasets
- Supporting business intelligence
- Enabling advanced analytics
- Ensuring data consistency across systems

19. Which is NOT a common data storage system?
    a. MySQL
    b. PostgreSQL
    c. MongoDB
    _d. Microsoft Word_

**Explanation**: Microsoft Word is a word processor, not a data storage system. Common data storage systems include:
- Relational Databases (MySQL, PostgreSQL)
  - Structured data storage
  - SQL query support
  - ACID compliance
- NoSQL Databases (MongoDB)
  - Flexible schema
  - Horizontal scaling
  - Various data models (document, key-value, etc.)
These systems are designed specifically for efficient data storage and retrieval.

20. What is the purpose of data warehousing?
    _a. To store integrated data for analysis_
    b. To collect new data
    c. To clean data
    d. To visualize data

**Explanation**: Data warehousing serves to:
- Centralize data from multiple sources
- Optimize data for analysis and reporting
- Maintain historical data
- Support business intelligence
- Enable complex analytical queries
Key characteristics include:
- Subject-oriented organization
- Integrated data from various sources
- Time-variant (historical) data
- Non-volatile storage
