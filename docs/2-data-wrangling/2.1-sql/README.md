# Data Querying with SQL: Your Gateway to Data Mastery

**After this submodule:** You can read and write portable SQL against relational data—from **SELECT** and filters through **JOIN**s, aggregations, and introductory analytics patterns.

## Why this matters

SQL is how most organizations **ask questions** of structured data at scale. Reports, dashboards, and many ML feature pipelines still pass through relational engines; fluent SQL lets you pull trustworthy slices without waiting on someone else for every tweak.

## Helpful video

High-level introduction to SQL and relational databases.

<iframe width="560" height="315" src="https://www.youtube.com/embed/27axs9dO7AE" title="What is SQL?" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Overview

**Prerequisites:** Comfortable with tables, rows, and columns (spreadsheet or [Pandas](../../1-data-fundamentals/1.5-data-analysis-pandas/README.md)). Install a client such as [DBeaver](../../0-prep/dbeaver.md) and a practice database as listed under **Tools Required** below.

> **Time needed:** Plan several hours across readings, the tutorial notebook, and practice.

## Lesson path (site order)

Work through these pages in order unless your instructor assigns otherwise:

1. [Intro to databases](intro-databases.md)  
2. [Basic operations](basic-operations.md)  
3. [Joins](joins.md)  
4. [Aggregations](aggregations.md)  
5. [Advanced concepts](advanced-concepts.md)  
6. [SQL project](project.md)  

Welcome to the fascinating world of SQL! Imagine having a conversation with your data - that's exactly what SQL allows you to do. Whether you're analyzing customer behavior, tracking business metrics, or uncovering hidden patterns, SQL is your trusted companion in the data journey.

Think of SQL as a universal language for data - just like how English helps people from different countries communicate, SQL helps different systems and people work with data in a standardized way. It's like having a Swiss Army knife for data manipulation: one tool that can slice, dice, filter, combine, and analyze data in countless ways.

For example, with a single SQL query, you can:

- Find your top 10 customers by revenue
- Calculate month-over-month growth rates
- Identify products frequently bought together
- Track user engagement patterns
- Generate complex business reports

And the best part? The same SQL query will work whether you're dealing with 100 records or 100 million records!

## Learning Objectives

By the end of this module, you will be able to:

1. Master the fundamentals of relational databases and SQL
   - Understand database architecture and design principles
   - Work with tables, schemas, and relationships
   - Handle different data types and constraints effectively

2. Craft elegant SQL queries from basic to advanced levels
   - Write clear, maintainable SELECT statements
   - Filter and sort data with precision
   - Perform calculations and data transformations
   - Master subqueries and CTEs

3. Apply industry-standard SQL best practices
   - Follow naming conventions and style guides
   - Write self-documenting code
   - Implement error handling
   - Ensure data integrity

4. Design complex data operations using joins and subqueries
   - Combine data from multiple tables efficiently
   - Write sophisticated nested queries
   - Use window functions for advanced analytics
   - Handle hierarchical data structures

5. Optimize queries for lightning-fast performance
   - Understand query execution plans
   - Use indexes effectively
   - Write efficient joins
   - Implement caching strategies

## Why SQL Matters

In today's data-driven world, SQL is more relevant than ever:

- **Universal Language**: SQL is the de facto standard for data manipulation
  - Used across industries and platforms
  - Consistent syntax and principles
  - Huge community and resources
  - Easy to learn, powerful to master

- **Career Essential**: 90% of Fortune 500 companies use SQL databases
  - Required skill for data analysts
  - Essential for business intelligence
  - Key for software development
  - Valuable for project management

- **Powerful Analysis**: Process millions of records in seconds
  - Efficient data processing
  - Complex calculations
  - Real-time analytics
  - Scalable solutions

- **Data Integration**: Connect and combine data from multiple sources
  - Merge data from different systems
  - Create unified views
  - Ensure data consistency
  - Enable cross-system analysis

Consider this real-world scenario:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
-- A single query that provides valuable business insights
SELECT 
    c.customer_segment,
    COUNT(DISTINCT o.customer_id) as num_customers,
    ROUND(AVG(o.total_amount), 2) as avg_order_value,
    SUM(o.total_amount) as total_revenue,
    COUNT(o.order_id) / COUNT(DISTINCT o.customer_id) as orders_per_customer
FROM orders o
JOIN customer_segments c ON o.customer_id = c.customer_id
WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY c.customer_segment
ORDER BY total_revenue DESC;

/* Output:
customer_segment | num_customers | avg_order_value | total_revenue | orders_per_customer
----------------|---------------|-----------------|---------------|--------------------
Premium         | 1250          | 185.50          | 580,937.50    | 2.5
Standard        | 3500          | 75.25           | 395,062.50    | 1.5
Basic           | 5250          | 45.75           | 240,187.50    | 1.0
*/
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-10" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Metrics and join</span>
    </div>
    <div class="code-callout__body">
      <p><strong>SELECT</strong> lists aggregates per segment after <strong>JOIN</strong> attaches each order to a segment. <code>COUNT(DISTINCT o.customer_id)</code> counts unique buyers; <code>AVG</code>/<code>SUM</code> summarize amounts; dividing order count by distinct customers approximates orders per buyer.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="11-20" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Filter, group, sort</span>
    </div>
    <div class="code-callout__body">
      <p><strong>WHERE</strong> restricts to recent orders. <strong>GROUP BY c.customer_segment</strong> produces one result row per segment. <strong>ORDER BY total_revenue DESC</strong> ranks segments; the comment block shows example output columns.</p>
    </div>
  </div>
</aside>
</div>

This single query tells us:

- Which customer segments are most valuable
- Average spending patterns
- Customer engagement levels
- Revenue distribution

## Module Overview

### 1. Introduction to Databases

{% include mermaid-diagram.html src="2-data-wrangling/2.1-sql/diagrams/README-1.mmd" %}

The diagram is intentionally high level: every topic below exists so that **queries return correct rows quickly** without letting bad writes corrupt other rows. Skim the bullets as a map; you will revisit each idea in the linked lessons.

Learn the building blocks of databases:

- **RDBMS Fundamentals**
  - PostgreSQL, MySQL, Oracle
  - Client-server architecture
  - ACID properties
  - Transaction management

- **Schema Design**
  - Normalization principles
  - Entity relationships
  - Data modeling best practices
  - Performance considerations

**Data Types & Constraints**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    price DECIMAL(10,2) CHECK (price >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    category_id INT REFERENCES categories(id)
);
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Table definition</span>
    </div>
    <div class="code-callout__body">
      <p><code>SERIAL PRIMARY KEY</code> auto-generates surrogate IDs. <code>NOT NULL</code> and <code>CHECK (price &gt;= 0)</code> reject bad rows at insert time. <code>REFERENCES categories(id)</code> enforces a foreign key so every product points at a real category; <code>DEFAULT CURRENT_TIMESTAMP</code> fills <code>created_at</code> automatically.</p>
    </div>
  </div>
</aside>
</div>

### 2. Basic SQL Operations

Master the fundamental operations with practical examples:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
-- Find top-selling products this month
SELECT 
    p.product_name,
    SUM(oi.quantity) as units_sold,
    SUM(oi.quantity * oi.price) as revenue,
    COUNT(DISTINCT o.customer_id) as unique_buyers
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_date >= DATE_TRUNC('month', CURRENT_DATE)
GROUP BY p.product_id, p.product_name
HAVING SUM(oi.quantity) > 0
ORDER BY units_sold DESC
LIMIT 5;

/* Output:
product_name    | units_sold | revenue  | unique_buyers
----------------|------------|----------|---------------
iPhone 13       | 250        | 250000   | 245
AirPods Pro     | 180        | 43200    | 178
MacBook Pro     | 75         | 150000   | 75
iPad Air        | 65         | 45500    | 63
Apple Watch     | 60         | 24000    | 58
*/
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-12" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Joins and filters</span>
    </div>
    <div class="code-callout__body">
      <p>Two <strong>JOIN</strong>s connect products → line items → orders. Aggregates (<code>SUM</code>, <code>COUNT DISTINCT</code>) are computed per product after <strong>GROUP BY p.product_id, p.product_name</strong>. <strong>HAVING</strong> drops groups with zero quantity sold.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-24" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Month filter and top five</span>
    </div>
    <div class="code-callout__body">
      <p><strong>WHERE</strong> uses <code>DATE_TRUNC('month', …)</code> so “this month” aligns to calendar boundaries. <strong>ORDER BY units_sold DESC</strong> ranks products; <strong>LIMIT 5</strong> returns only the leaders. The comment shows example numeric output.</p>
    </div>
  </div>
</aside>
</div>

Notice how **JOIN** links products to line items and orders, **WHERE** limits to the current month, **GROUP BY** rolls up to each product, and **ORDER BY** with **LIMIT** surfaces the top sellers—one pipeline from fact tables to a ranking.

### 3. Aggregations and Grouping

Transform raw data into actionable insights:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
-- Customer cohort analysis
WITH cohort_data AS (
    SELECT 
        DATE_TRUNC('month', first_order_date) as cohort_month,
        COUNT(DISTINCT customer_id) as cohort_size,
        SUM(total_spent) as total_revenue,
        AVG(orders_count) as avg_orders
    FROM customer_metrics
    GROUP BY DATE_TRUNC('month', first_order_date)
)
SELECT 
    cohort_month,
    cohort_size,
    ROUND(total_revenue/cohort_size, 2) as revenue_per_customer,
    ROUND(avg_orders, 1) as avg_orders_per_customer
FROM cohort_data
ORDER BY cohort_month DESC;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CTE: cohort aggregates</span>
    </div>
    <div class="code-callout__body">
      <p>The <strong>WITH</strong> clause builds <code>cohort_data</code>: one row per month of first order. <code>DATE_TRUNC('month', first_order_date)</code> buckets customers; inner <strong>SELECT</strong> computes cohort size, revenue, and average orders from <code>customer_metrics</code>.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="9-17" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Outer SELECT</span>
    </div>
    <div class="code-callout__body">
      <p>The outer query only reshapes CTE rows: revenue per customer, rounded averages, and <strong>ORDER BY cohort_month DESC</strong> so the newest cohorts appear first—no extra joins needed.</p>
    </div>
  </div>
</aside>
</div>

The **CTE** (`WITH`) names the grouped cohort metrics first; the outer query only formats and sorts. That pattern keeps long aggregate queries readable when you add more cohort dimensions later.

Performance consideration:
$T_{query} = O(n \log n)$ for sorted aggregations
$T_{memory} = O(k)$ where k = number of groups

### 4. Joins and Relationships

Master the art of combining data:

{% include mermaid-diagram.html src="2-data-wrangling/2.1-sql/diagrams/README-2.mmd" %}

Example of complex joins:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
-- Customer order history with product details
WITH customer_orders AS (
    SELECT 
        c.customer_id,
        c.name as customer_name,
        o.order_id,
        o.order_date,
        p.product_name,
        oi.quantity,
        oi.price as unit_price,
        oi.quantity * oi.price as total_amount
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    LEFT JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN products p ON oi.product_id = p.product_id
)
SELECT 
    customer_name,
    COUNT(DISTINCT order_id) as total_orders,
    SUM(total_amount) as total_spent,
    STRING_AGG(DISTINCT product_name, ', ') as products_bought
FROM customer_orders
GROUP BY customer_id, customer_name
HAVING COUNT(DISTINCT order_id) > 0;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-12" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Detail rows (CTE)</span>
    </div>
    <div class="code-callout__body">
      <p><code>customer_orders</code> flattens customers to orders to line items to products. Chained <strong>LEFT JOIN</strong>s preserve customers with no orders (NULLs on the right). <code>quantity * … price</code> is line revenue for downstream aggregation.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="13-24" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Roll up per customer</span>
    </div>
    <div class="code-callout__body">
      <p>The outer <strong>SELECT</strong> groups by customer, counts distinct orders, sums revenue, and <code>STRING_AGG</code> lists product names. <strong>HAVING</strong> removes customers who never placed an order (all-null joins).</p>
    </div>
  </div>
</aside>
</div>

**Reading the result:** chained **LEFT JOIN**s keep every customer even with sparse orders; `STRING_AGG` rolls product names into one line per customer; **`HAVING`** drops customers with no orders after the join. Compare with **INNER JOIN** if you only want buyers.

### 5. Advanced SQL Concepts

Take your SQL skills to the next level:

**1. Query optimization**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
EXPLAIN ANALYZE
SELECT /*+ INDEX(orders idx_order_date) */
    DATE_TRUNC('month', order_date) as month,
    COUNT(*) as order_count,
    SUM(total_amount) as revenue
FROM orders
WHERE order_date >= CURRENT_DATE - INTERVAL '1 year'
GROUP BY DATE_TRUNC('month', order_date);
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Plan and monthly revenue</span>
    </div>
    <div class="code-callout__body">
      <p><strong>EXPLAIN ANALYZE</strong> runs the query and prints the executor plan plus actual timings—use it to see index use vs sequential scans. The <strong>SELECT</strong> buckets <code>order_date</code> by month over the last year and aggregates order counts and revenue (hint comment suggests an index on <code>order_date</code>).</p>
    </div>
  </div>
</aside>
</div>

**2. Window functions**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
SELECT 
    category_name,
    product_name,
    price,
    AVG(price) OVER (PARTITION BY category_name) as avg_category_price,
    price - AVG(price) OVER (PARTITION BY category_name) as price_diff_from_avg,
    RANK() OVER (PARTITION BY category_name ORDER BY price DESC) as price_rank
FROM products p
JOIN categories c ON p.category_id = c.category_id;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-9" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Partitioned windows</span>
    </div>
    <div class="code-callout__body">
      <p><code>AVG(price) OVER (PARTITION BY category_name)</code> computes each row’s category average without collapsing rows. The next column subtracts that average from price (distance from typical). <code>RANK() … ORDER BY price DESC</code> ranks items inside each category.</p>
    </div>
  </div>
</aside>
</div>

**3. Common table expressions (CTEs)**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
WITH RECURSIVE subordinates AS (
    -- Base case: direct reports
    SELECT employee_id, manager_id, name, 1 as level
    FROM employees
    WHERE manager_id = 1
    
    UNION ALL
    
    -- Recursive case: subordinates of subordinates
    SELECT e.employee_id, e.manager_id, e.name, s.level + 1
    FROM employees e
    JOIN subordinates s ON e.manager_id = s.employee_id
)
SELECT * FROM subordinates ORDER BY level, employee_id;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-14" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Recursive org chart</span>
    </div>
    <div class="code-callout__body">
      <p><strong>WITH RECURSIVE</strong> seeds with direct reports to manager 1, then <strong>UNION ALL</strong> joins employees to the growing result so each iteration walks one level down the tree. <code>level</code> increments until no new rows; the final <strong>SELECT</strong> returns the full subtree sorted by depth and id.</p>
    </div>
  </div>
</aside>
</div>

Performance considerations:

- Query Cost = $I/O + CPU + Memory$
- Index Usage = $\frac{SelectivityFactor \times DataSize}{IndexSize}$
- Join Cost = $O(n \log n)$ for hash joins

## Prerequisites

Before starting this journey, ensure you have:

1. **Basic Understanding of Data Structures**
   - Arrays and Lists: How data is organized sequentially
   - Key-Value Pairs: Understanding relationships between data points
   - Trees and Graphs: Hierarchical data organization
   - Basic Set Theory: Union, intersection, difference operations

2. **Familiarity with Database Concepts**
   - Data Organization: Tables, rows, and columns
   - Basic CRUD Operations: Create, Read, Update, Delete
   - Understanding of Tables and Relationships
   - Basic Data Types: Numbers, text, dates

3. **Development Environment**
   - PostgreSQL 13+ installed
   - Basic command line familiarity
   - Text editor for SQL scripts
   - Git for version control (optional)

4. **Mathematical Foundation**
   - Basic arithmetic operations
   - Percentage calculations
   - Simple statistics (average, sum, count)
   - Basic logical operations

## Tools Required

**1. Online SQL Compilers (no installation needed)**

If you prefer to practice SQL directly in the browser without installing anything:

- **SQLite Online** — [sqliteonline.com](https://sqliteonline.com/) — run SQL queries instantly in the browser
- **DB Browser for SQLite** — [sqlitebrowser.org](https://sqlitebrowser.org/) — desktop GUI for SQLite databases

**2. DBeaver Community Edition**

- Universal Database Tool

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight bash %}
# Installation commands
# For macOS:
brew install --cask dbeaver-community

# For Ubuntu:
sudo snap install dbeaver-ce
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-6" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Install DBeaver</span>
    </div>
    <div class="code-callout__body">
      <p><strong>Homebrew</strong> (macOS) and <strong>snap</strong> (Ubuntu) install the community edition GUI. After install, create a connection to your Postgres (or other) instance and use the SQL editor to run the course snippets.</p>
    </div>
  </div>
</aside>
</div>

Features:

- SQL Editor with syntax highlighting
- Visual Query Builder
- ERD (Entity Relationship Diagram) viewer
- Data export/import wizards
- Multi-platform support (Windows, macOS, Linux)
- Connection templates for all major databases

**2. Sample database (Northwind)**

Real-world business scenario database including:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
-- Core tables
- Customers (customer demographics and contacts)
- Products (product details and inventory)
- Orders (order details and shipping)
- Employees (staff information and territories)

-- Key relationships
- One customer can have many orders
- One order can have multiple products
- Each product belongs to a category
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-10" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Northwind outline</span>
    </div>
    <div class="code-callout__body">
      <p>This block is a **schema sketch** (comments, not runnable DDL): core entities are customers, products, orders, and employees, with classic one-to-many paths (customer → orders → line items) and categories on products.</p>
    </div>
  </div>
</aside>
</div>

Installation:

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
-- 1. Create database
CREATE DATABASE northwind;

-- 2. Import schema
psql -d northwind -f northwind_schema.sql

-- 3. Import data
psql -d northwind -f northwind_data.sql
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Create DB and load dumps</span>
    </div>
    <div class="code-callout__body">
      <p><strong>CREATE DATABASE</strong> makes an empty database. <code>psql -f</code> runs SQL files: first schema (tables, keys), then data. Paths to the dump files must match your download; run from a shell where <code>psql</code> is on <code>PATH</code>.</p>
    </div>
  </div>
</aside>
</div>

**3. Additional tools (optional)**

- **pgAdmin 4**: Alternative GUI for PostgreSQL
- **Visual Studio Code**: With SQL extensions
- **DataGrip**: JetBrains SQL IDE (paid)
- **Postman**: For testing database APIs

**4. Version control setup**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight bash %}
# Initialize SQL project
mkdir sql-practice
cd sql-practice
git init

# Create .gitignore
echo "*.log" > .gitignore
echo "*.tmp" >> .gitignore
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Git folder for scripts</span>
    </div>
    <div class="code-callout__body">
      <p>Creates a directory and <strong>git init</strong> for versioned <code>.sql</code> files. <code>.gitignore</code> excludes logs and temp files so accidental local artifacts do not get committed.</p>
    </div>
  </div>
</aside>
</div>

## Best Practices

**1. Query writing standards**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
--  Bad Practice
SELECT * FROM orders o, customers c WHERE o.customer_id=c.id;

--  Good Practice
SELECT 
    c.first_name,
    c.last_name,
    o.order_date,
    o.total_amount
FROM orders o
JOIN customers c 
    ON o.customer_id = c.id
WHERE 
    o.order_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY 
    o.order_date DESC;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Implicit join (avoid)</span>
    </div>
    <div class="code-callout__body">
      <p>The “comma join” with <strong>WHERE</strong> is easy to misread and easy to turn into a Cartesian product if you forget a predicate. Prefer explicit <strong>JOIN … ON</strong> so relationships stay visible in the <strong>FROM</strong> clause.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="9-16" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Explicit JOIN and columns</span>
    </div>
    <div class="code-callout__body">
      <p>Lists only needed columns, joins orders to customers on <code>customer_id</code>, filters recent orders, and orders by date—readable structure for reviewers and for the optimizer.</p>
    </div>
  </div>
</aside>
</div>

**2. Performance optimization**

**Indexing strategy**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
-- Create indexes for frequently queried columns
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_orders_customer ON orders(customer_id);

-- Use composite indexes for common query patterns
CREATE INDEX idx_orders_customer_date 
ON orders(customer_id, order_date);
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-7" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Single- and multi-column indexes</span>
    </div>
    <div class="code-callout__body">
      <p>B-tree indexes on <code>order_date</code> and <code>customer_id</code> speed filters and joins. The composite index matches queries that filter by customer <em>and</em> date—column order should match your most selective predicates.</p>
    </div>
  </div>
</aside>
</div>

**Query optimization**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
--  Bad: Full table scan
SELECT * FROM orders 
WHERE EXTRACT(YEAR FROM order_date) = 2023;

--  Good: Uses index
SELECT * FROM orders 
WHERE order_date >= '2023-01-01' 
  AND order_date < '2024-01-01';
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Sargable date range</span>
    </div>
    <div class="code-callout__body">
      <p><code>EXTRACT(YEAR FROM order_date) = 2023</code> applies a function to the column, which often blocks index use. Prefer a **range** on <code>order_date</code> (<code>&gt;=</code> start and <code>&lt;</code> end of next year) so the planner can use a btree index.</p>
    </div>
  </div>
</aside>
</div>

**3. Data integrity**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
-- Use constraints to enforce business rules
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    price DECIMAL(10,2) CHECK (price >= 0),
    stock INT CHECK (stock >= 0),
    category_id INT REFERENCES categories(id)
);
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-8" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Constraints</span>
    </div>
    <div class="code-callout__body">
      <p><code>CHECK</code> clauses forbid negative price or stock. <code>REFERENCES categories(id)</code> ties each product to an existing category row. Together they reject bad inserts before application code sees them.</p>
    </div>
  </div>
</aside>
</div>

**4. Code organization**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight sql %}
-- Use CTEs for complex queries
WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', order_date) as month,
        SUM(total_amount) as revenue
    FROM orders
    GROUP BY DATE_TRUNC('month', order_date)
),
sales_growth AS (
    SELECT 
        month,
        revenue,
        LAG(revenue) OVER (ORDER BY month) as prev_month_revenue
    FROM monthly_sales
)
SELECT 
    month,
    revenue,
    ROUND(
        ((revenue - prev_month_revenue) / prev_month_revenue * 100)::numeric, 
        2
    ) as growth_rate
FROM sales_growth;
{% endhighlight %}
</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-11" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Monthly revenue CTE</span>
    </div>
    <div class="code-callout__body">
      <p>First CTE aggregates orders to one revenue row per calendar month. Second CTE uses <code>LAG(revenue) OVER (ORDER BY month)</code> to pull the previous month’s revenue on the same row—setup for period-over-period math.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="12-23" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Growth rate</span>
    </div>
    <div class="code-callout__body">
      <p>Final <strong>SELECT</strong> computes percent change vs prior month; the <code>::numeric</code> cast and <code>ROUND</code> control display. Guard division-by-null if the first month has no <code>LAG</code> (not shown here).</p>
    </div>
  </div>
</aside>
</div>

## Resources

### Official Documentation

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
  - Complete reference for PostgreSQL
  - Detailed explanations and examples
  - Performance tuning guidelines
  - Security best practices

- [SQL Style Guide](https://www.sqlstyle.guide/)
  - Industry-standard formatting
  - Naming conventions
  - Code organization
  - Documentation practices

### Interactive Learning

1. **Practice Platforms**
   - **LeetCode SQL Path**
     - 50+ SQL problems
     - Difficulty progression
     - Real interview questions

   - **HackerRank SQL Track**
     - Basic to advanced challenges
     - Instant feedback
     - Certification available

   - **SQL Zoo**
     - Interactive tutorials
     - Progressive learning
     - Real-world examples

2. **Online Courses**
   - **Stanford's Database Course**
     - Comprehensive coverage
     - Academic perspective
     - Free access

   - **Mode Analytics SQL Tutorial**
     - Business-focused examples
     - Interactive exercises
     - Real data scenarios

### Essential Books

1. **For Beginners**
   - "Learning SQL" by Alan Beaulieu
   - "SQL Queries for Mere Mortals"
   - "Head First SQL"

2. **For Advanced Users**
   - "SQL Performance Explained" by Markus Winand
   - "SQL Antipatterns" by Bill Karwin
   - "High Performance SQL" by Baron Schwartz

3. **Specialized Topics**
   - "PostgreSQL: Up and Running"
   - "Database Design for Mere Mortals"
   - "Data Analysis Using SQL and Excel"

### Community Resources

1. **Forums & Communities**
   - Stack Overflow SQL Tag
   - PostgreSQL Mailing Lists
   - Reddit r/SQL
   - Database Administrators Stack Exchange

2. **Blogs & Newsletters**
   - Use The Index, Luke!
   - Planet PostgreSQL
   - SQLBlog.org
   - Weekly SQL Newsletter

3. **Tools & Utilities**
   - SQLFormat.org (Query formatter)
   - DbDiagram.io (Database design)
   - SQLFiddle (Query testing)
   - Explain.depesz.com (Query plan analysis)

## Assignment

Ready to test your SQL skills? Head over to the [Module 2 assignment (student version)](../assignments/module-assignment-student.md) to apply what you have learned.

## What's Next?

Get ready to embark on an exciting journey into the world of data querying! We'll start with the basics and gradually move to advanced concepts, with plenty of hands-on exercises along the way.

Remember: "Data is the new oil, and SQL is the drill!"

Let's dive in and master SQL together!
