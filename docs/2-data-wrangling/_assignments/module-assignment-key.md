# Data Wrangling Module Assessment - Answer Key

## SQL (25 points)

1. Which SQL command is used to retrieve data from a database?
   a. **SELECT**
   b. UPDATE
   c. INSERT
   d. DELETE

   **Explanation**: SELECT is the fundamental command for querying data. While other commands modify data (UPDATE, INSERT, DELETE), SELECT is specifically for data retrieval.

2. What is the purpose of the WHERE clause?
   a. To sort data
   b. To group data
   c. **To filter data**
   d. To join tables

   **Explanation**: The WHERE clause filters rows based on specified conditions. It's applied before grouping (GROUP BY) and sorting (ORDER BY).

3. Which JOIN type returns all records from both tables?
   a. INNER JOIN
   b. **FULL OUTER JOIN**
   c. LEFT JOIN
   d. RIGHT JOIN

   **Explanation**: FULL OUTER JOIN returns all records from both tables, with NULL values where there are no matches. Other JOIN types may exclude records:
   - INNER JOIN: Only matching records
   - LEFT JOIN: All records from left table
   - RIGHT JOIN: All records from right table

4. What is the difference between HAVING and WHERE?
   a. **HAVING filters grouped results, WHERE filters individual rows**
   b. HAVING is used before GROUP BY, WHERE after
   c. They are interchangeable
   d. HAVING can only be used with JOINs

   **Explanation**: HAVING is used to filter groups after GROUP BY, while WHERE filters individual rows before grouping. This is a fundamental difference in SQL query processing order.

5. Which statement about indexes is correct?
   a. Indexes always improve query performance
   b. **Indexes can slow down INSERT and UPDATE operations**
   c. You should index every column
   d. Indexes don't affect storage space

   **Explanation**: While indexes speed up SELECT queries, they require additional storage space and maintenance during INSERT and UPDATE operations, as the index must be updated along with the data.

## Data Wrangling (25 points)

6. What is data cleaning?
   a. Collecting data from various sources
   b. Analyzing data patterns and trends
   c. **Fixing or removing incorrect, corrupted, or irrelevant data**
   d. Visualizing data relationships

   **Explanation**: Data cleaning is specifically about improving data quality by addressing issues like errors, inconsistencies, and irrelevant information. It's a crucial step before analysis.

7. Which of these is NOT a common data quality issue?
   a. Missing values in customer records
   b. Duplicate transaction entries
   c. Inconsistent date formats
   d. **Beautiful data visualization**

   **Explanation**: Data visualization is about presenting data, not a quality issue. The other options are genuine data quality problems that need to be addressed during data cleaning.

8. What is the best approach for handling missing values?
   a. Always delete rows with missing values
   b. Always replace with zero
   c. **Choose method based on context and data patterns**
   d. Ignore them completely

   **Explanation**: The appropriate method depends on factors like:
   - The amount of missing data
   - The pattern of missingness
   - The importance of the missing information
   - The potential impact on analysis

9. What is data transformation?
   a. **Converting data from one format or structure to another**
   b. Deleting incorrect data entries
   c. Collecting new data samples
   d. Analyzing data patterns

   **Explanation**: Data transformation involves changing data format or structure to make it more suitable for analysis, including:
   - Scaling numerical features
   - Encoding categorical variables
   - Feature engineering
   - Format standardization

10. Which scaling method is most robust to outliers?
    a. Standard scaling (z-score)
    b. **Robust scaling (using quartiles)**
    c. Min-max scaling
    d. Log transformation

    **Explanation**: Robust scaling uses statistics that are not influenced by outliers:
    - Uses median instead of mean
    - Uses interquartile range instead of standard deviation
    - Maintains relative relationships while reducing outlier impact

## Exploratory Data Analysis (25 points)

11. What is the main purpose of EDA?
    a. **To understand patterns and relationships in data**
    b. To clean data
    c. To collect data
    d. To make predictions

    **Explanation**: EDA is primarily about understanding your data before formal modeling. While data cleaning might occur during EDA, the main goal is to discover patterns, spot anomalies, test hypotheses, and form insights.

12. Which plot is best for showing the distribution of a continuous variable?
    a. Bar plot
    b. Scatter plot
    c. **Histogram**
    d. Pie chart

    **Explanation**: Histograms are ideal for continuous variables because they:
    - Show the shape of the distribution
    - Reveal the spread and central tendency
    - Help identify outliers and gaps in the data

13. What does a box plot show?
    a. Only the median
    b. **Five-number summary and outliers**
    c. Just the outliers
    d. Mean and standard deviation

    **Explanation**: A box plot shows:
    - Median (middle line)
    - Q1 and Q3 (box edges)
    - Whiskers (typically 1.5 * IQR)
    - Individual points for outliers

14. Which correlation coefficient ranges from -1 to 1?
    a. **Pearson**
    b. Chi-square
    c. F-statistic
    d. T-statistic

    **Explanation**: The Pearson correlation coefficient:
    - Ranges from -1 (perfect negative correlation) to +1 (perfect positive correlation)
    - 0 indicates no linear correlation
    - Measures the strength and direction of linear relationships

15. What type of plot is best for showing relationships between two continuous variables?
    a. **Scatter plot**
    b. Bar plot
    c. Line plot
    d. Pie chart

    **Explanation**: Scatter plots are ideal for showing relationships between continuous variables because they:
    - Show the pattern of association
    - Reveal the strength of relationship
    - Help identify outliers
    - Can be enhanced with trend lines

## Data Engineering (25 points)

16. What is ETL in data engineering?
    a. **Extract, Transform, Load**
    b. Export, Transfer, Link
    c. Evaluate, Test, Launch
    d. Extract, Test, Load

    **Explanation**: ETL is a fundamental data pipeline process where:
    - Data is extracted from source systems
    - Transformed to meet business needs
    - Loaded into target systems

17. Which storage solution is most appropriate for structured, relational data with ACID requirements?
    a. Text files
    b. **Relational databases**
    c. PDF files
    d. Image files

    **Explanation**: Relational databases are ideal for structured data because they:
    - Enforce data integrity
    - Support complex queries
    - Provide ACID compliance
    - Enable efficient indexing

18. What is the primary purpose of data integration?
    a. **Combining data from different sources**
    b. Cleaning data
    c. Analyzing data
    d. Visualizing data

    **Explanation**: Data integration involves:
    - Merging multiple data sources
    - Resolving schema conflicts
    - Ensuring data consistency
    - Creating unified views

19. Which is NOT typically used as a data storage system?
    a. MySQL
    b. PostgreSQL
    c. MongoDB
    d. **Microsoft Word**

    **Explanation**: While MySQL, PostgreSQL, and MongoDB are purpose-built databases:
    - MySQL: Relational database
    - PostgreSQL: Advanced relational database
    - MongoDB: NoSQL document store
    - Microsoft Word: Word processor, not designed for data storage

20. What is the primary purpose of data warehousing?
    a. **To store integrated data for analysis**
    b. To collect new data
    c. To clean data
    d. To visualize data

    **Explanation**: Data warehouses are designed to:
    - Centralize enterprise data
    - Support complex analytics
    - Maintain historical records
    - Enable business intelligence
