# Mastering Basic SQL Operations: Your Data Query Journey

**After this lesson:** You can **CREATE** tables, **INSERT** rows, **SELECT** and filter data with **WHERE**, **UPDATE** and **DELETE** safely, and read simple query plans.

## Helpful video

High-level introduction to SQL and relational databases.

## Overview

**Prerequisites:** [Introduction to Databases](intro-databases.md) (tables, keys, types). Have a SQL client and a practice database as described in the [module README](./).

> **Time needed:** About 60-90 minutes with hands-on practice.

## Why this matters

**CREATE**, **READ**, **UPDATE**, and **DELETE** are the daily loop of working with data: define structure, pull slices for analysis, correct mistakes, and retire bad rows safely. Aggregations, joins, and window functions all assume you are fluent here first.

## Introduction to SQL Basics

SQL (Structured Query Language) is the standard language for managing and manipulating relational databases. Understanding basic SQL operations is important for:

* Data retrieval and analysis
* Database management
* Data integrity maintenance
* Application development

## CRUD Operations

The sections below follow the same order most people learn: define a table, put rows in, read them back, then update or delete with a **scoped** `WHERE` so you do not touch the whole table by accident.

> **Warning:** Always include a `WHERE` clause with `UPDATE` and `DELETE`. Without one, the operation applies to **every row** in the table. Test your `WHERE` condition with a `SELECT` first.

### 1. CREATE: Adding Data

CREATE TABLE with constraints

`SERIAL PRIMARY KEY` auto-increments the ID. `NOT NULL` rejects missing names. `UNIQUE` prevents duplicate emails. `DEFAULT CURRENT_TIMESTAMP` records the signup time automatically without requiring the caller to supply it.

INSERT: single and multi-row

The first `INSERT` adds one row by listing column names then values. The second uses a single statement with multiple value tuples, more efficient than separate inserts because it sends one round-trip to the database.

### 2. READ: Querying Data

SELECT \* and specific columns

`SELECT *` returns every column, convenient for exploration but avoid in production as schema changes can break downstream code. Listing columns explicitly (`first_name, last_name, email`) makes the contract clear.

WHERE filtering and LIKE pattern

**WHERE last\_name = 'Smith'** does an exact match. `LIKE '%@email.com'` uses a wildcard to match any email ending with that domain, useful for finding customers from a specific provider.

### 3. UPDATE: Modifying Data

Targeted UPDATE with WHERE

Always scope an `UPDATE` with a `WHERE` clause. Without it, every row is changed. The first example corrects one customer's email; the second fills missing `created_at` values on all rows where it's NULL.

Conditional UPDATE with INITCAP

`INITCAP` title-cases a string. The `WHERE` clause only touches rows where at least one name is already in the wrong case, avoiding unnecessary writes on rows that are already correct.

### 4. DELETE: Removing Data

DELETE a specific row by primary key

Scoping `DELETE` by primary key is the safest approach, exactly one row is removed. Without `WHERE` the entire table would be cleared.

Delete rows matching a condition

Deletes all customers whose account was created more than a year ago, a typical data-retention policy. The interval comparison is evaluated at runtime so the cutoff shifts with calendar time.

TRUNCATE for bulk removal

`TRUNCATE` removes all rows faster than `DELETE` without a `WHERE` because it skips row-by-row logging. Use it when you need to empty a table entirely, e.g. resetting a staging table before a reload.

## Basic Query Structure

### 1. SELECT Statement Anatomy

Columns: aliases and derived values

The `SELECT` list defines output columns. `AS alias` renames a column in results. `CONCAT(...)` creates a derived column combining two source columns, no schema change needed.

Clauses in execution order

SQL processes clauses in this order: **FROM** → **WHERE** → **GROUP BY** → **HAVING** → **SELECT** → **ORDER BY** → **LIMIT**. The written order differs from the execution order, that matters when diagnosing unexpected results.

### 2. Filtering and Sorting

WHERE with AND, IN, and BETWEEN

Multiple `AND` conditions narrow results to rows meeting all criteria. `IN ('pending', 'processing')` is cleaner than chained `OR`. `BETWEEN … AND …` is inclusive on both ends, combine it with `CURRENT_DATE` for rolling windows.

LIKE, ILIKE, and ORDER BY

`LIKE '%.com'` matches case-sensitively; `ILIKE 'j%'` is case-insensitive (PostgreSQL extension). The final `ORDER BY price DESC, product_name ASC` sorts by two columns, price descending, then alphabetically within the same price.

## Data Types and Constraints

### 1. Common Data Types

Numeric and text columns

`SERIAL` auto-increments integers for surrogate keys. `DECIMAL(10,2)` stores exact money values. `VARCHAR(n)` caps string length; `TEXT` is unlimited, use it for user-written content.

Date/time, boolean, and enum columns

`TIMESTAMP` stores full date+time; `DATE` stores date only. `BOOLEAN` holds true/false flags. Custom enum types like `product_status` constrain a column to a known set of string values, better than free-text strings for state columns.

### 2. Constraints

Primary key and foreign key constraints

`PRIMARY KEY` uniquely identifies each order. `REFERENCES customers(customer_id)` is a foreign key, the database rejects inserts where the customer\_id doesn't exist in the customers table, keeping referential integrity.

NOT NULL, UNIQUE, CHECK, and DEFAULT

`NOT NULL` prevents missing order dates. `UNIQUE` ensures no two orders share a tracking number. `CHECK (total_amount >= 0)` rejects negative totals at insert time. `DEFAULT 'pending'` sets the initial status without requiring the caller to supply it.

## Table Relationships

### 1. One-to-Many Relationship

Parent table: categories

The `categories` table is the "one" side. Its `category_id` primary key will be referenced by the products table.

Child table: products with foreign key

`REFERENCES categories(category_id)` creates a one-to-many link: one category can have many products, but each product belongs to at most one category. The database enforces this, inserting a product with an unknown category\_id fails.

### 2. Many-to-Many Relationship

Independent parent tables

`products` and `orders` each have their own primary keys. Neither references the other directly, the relationship is expressed through the junction table below.

Junction table with composite primary key

`order_items` is the "many-to-many" bridge: one order can contain many products and one product can appear in many orders. `PRIMARY KEY (order_id, product_id)` prevents duplicate line-item pairs. It also stores line-level attributes (`quantity`, `price_at_time`) that belong to the association, not to either parent.

## Basic Joins

### 1. INNER JOIN

INNER JOIN: only matched rows

`INNER JOIN customers c ON o.customer_id = c.customer_id` returns only orders that have a matching customer, orders with a missing or invalid customer\_id are dropped. Use an alias (`o`, `c`) to keep column references short and unambiguous.

### 2. LEFT JOIN

LEFT JOIN: all customers including those with no orders

`LEFT JOIN orders` keeps every customer row even if there are no matching orders. `COALESCE(SUM(...), 0)` replaces the NULL total that results from a non-matching left-join row with zero, so customers who never ordered show `0` rather than `NULL`.

### 3. Multiple Joins

Three-table chain join for line-item detail

Chaining four tables (`orders → customers`, `orders → order_items`, `order_items → products`) flattens the schema into one row per line item. `oi.quantity * oi.price_at_time` computes the line total. `ORDER BY o.order_id, p.name` groups line items by order then sorts products alphabetically within each order.

## Additional Real-World Business Scenarios

### 1. E-commerce Order Analytics

CTE: daily order metrics with new-customer detection

`order_metrics` groups by day over the last 30 days. The `COUNT(DISTINCT CASE WHEN customer_id NOT IN (...))` subquery identifies first-time buyers by checking whether a customer placed any order before today, an anti-join pattern inline.

Outer query: derived KPIs

Computes revenue per customer, new-customer percentage, and average order value from the CTE columns. All numeric results use `ROUND(::numeric, 2)` for consistent decimal output. Sorted newest day first.

### 2. Customer Segmentation

CTE: aggregate metrics per customer

`customer_metrics` joins customers → orders with `LEFT JOIN` to retain customers with no orders. Aggregates per customer: order count, total and average spend, first/last order dates, and active-month count.

Recency and value segment labels

The first `CASE` computes a recency label (Never Ordered / Active / At Risk / Churned) based on days since last order. The second computes a value segment (VIP / Regular / New / Inactive) from spend and order frequency thresholds.

### 3. Product Performance

CTE: product sales and review aggregates

`product_metrics` chains four `LEFT JOIN`s (order items → orders → reviews) to keep products with no sales or reviews. Aggregates cover order count, units sold, revenue, average rating, and review count per product.

Outer query: derived pricing and unit metrics

Computes average selling price (`revenue / units_sold`) and units per customer from the CTE. `NULLIF` guards both divisions. Sorted by revenue descending so the top-earning products appear first.

## Performance Optimization Examples

### 1. Index Usage

Create indexes for join and filter columns

The composite index on `(customer_id, order_date DESC)` speeds up queries that filter by customer and sort by date. The covering index with `INCLUDE` avoids a table heap lookup when the query only needs those columns.

EXPLAIN ANALYZE to verify index use

`EXPLAIN ANALYZE` runs the query and prints the actual execution plan. Check whether the index created above shows up as an index scan (fast) instead of a sequential scan (slow), and compare planned vs. actual row counts.

### 2. Query Optimization

Slow: IN with a subquery

`WHERE customer_id IN (SELECT ...)` can force the engine to materialise the subquery and then probe it for each order row, inefficient on large tables without an index on the subquery column.

Better: JOIN replaces the subquery

A `JOIN` lets the planner choose an optimal hash or merge join, often far faster than the `IN` subquery. `EXISTS` (shown last) is a further alternative, it short-circuits as soon as one matching row is found.

### 3. Batch Processing

PL/pgSQL block setup

The anonymous `DO $$` block declares variables for batch size and counters. The outer `LOOP` iterates until no unprocessed rows remain.

CTE + UPDATE with FOR UPDATE SKIP LOCKED

The CTE selects the next batch of unprocessed rows. `FOR UPDATE SKIP LOCKED` locks only the selected rows and skips any already locked by another session, safe for concurrent workers. The `UPDATE` marks the batch as processed.

Progress tracking and loop exit

`GET DIAGNOSTICS batch_count = ROW_COUNT` captures how many rows the last statement affected. `EXIT WHEN batch_count = 0` stops the loop when no rows remain. `COMMIT` after each batch keeps transactions small and releases locks early.

## Common Pitfalls and Solutions

### 1. N+1 Query Problem

N+1 problem: correlated subquery per row

The scalar subquery inside `SELECT` runs once for every row in `orders`. With 10,000 orders that's 10,001 round-trips to the database, slow and unscalable.

Fix: single JOIN fetches all rows at once

Replacing the subquery with a `JOIN` reduces the query to a single pass. The planner can choose an efficient hash or merge join strategy instead of looping.

### 2. Cartesian Products

Bad: implicit comma join hides the condition

The comma between table names produces a full Cartesian product first; the `WHERE` clause filters it after. Omitting or mistyping the condition silently returns every combination of rows.

Good: explicit JOIN keeps intent visible

The explicit `JOIN … ON` syntax makes the join condition part of the join itself, not a filter afterthought. Missing or wrong `ON` clauses are a syntax or logical error, not a silent blowup.

### 3. NULL Handling

Bad: = NULL always returns NULL (no rows)

In SQL, `NULL = NULL` evaluates to `NULL`, not `TRUE`. A `WHERE price = NULL` predicate silently returns zero rows, no error, no warning.

Good: IS NULL checks for absence of a value

`IS NULL` is the correct predicate for checking missing values. It returns rows where the column has no value at all.

Better: COALESCE substitutes a default

`COALESCE(price, 0)` returns the first non-NULL argument. This is useful in SELECT lists to present NULLs as a meaningful default (0 for prices, a placeholder string for descriptions) without altering the stored data.

## Best Practices Checklist

1. **Query Structure**
   * Use meaningful table aliases
   * Format queries for readability
   * Comment complex logic
   * Use CTEs for better organization
2. **Performance**
   * Create appropriate indexes
   * Filter early in the query
   * Avoid SELECT \*
   * Use EXPLAIN ANALYZE
3. **Data Quality**
   * Handle NULL values appropriately
   * Validate input data
   * Use constraints
   * Implement error handling
4. **Maintenance**
   * Document queries
   * Use version control
   * Monitor performance
   * Regular optimization

Remember: "Clean, efficient queries lead to better performance and maintainability!"

## Gotchas

* **Running UPDATE or DELETE without a WHERE clause**: Without a filter, every row in the table is modified or removed. Before writing a destructive statement, run the equivalent SELECT with the same WHERE to confirm the affected rows, then substitute UPDATE/DELETE.
* **TRUNCATE is not the same as DELETE with no WHERE**: `TRUNCATE TABLE` is much faster but cannot be rolled back in some databases (and in PostgreSQL it can be, but it bypasses row-level triggers and resets sequences). Use `DELETE` when you need triggers to fire or when a partial rollback is possible.
* **Relying on the written order of SQL clauses as the execution order**: SQL is processed as FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT. That means a column alias defined in SELECT is not yet available in WHERE or HAVING; reference the original expression instead.
* **Using LIKE with a leading wildcard and expecting fast queries**: `LIKE '%@email.com'` cannot use a standard B-tree index because the pattern starts with a wildcard; the database must scan every row. Anchor patterns to the left (`LIKE 'john%'`) or add a full-text or trigram index for suffix searches.
* **Assuming COUNT(\*) and COUNT(column) are interchangeable**, `COUNT(*)` counts all rows including those with NULLs; `COUNT(column)` skips rows where that column is NULL. Using the wrong form in an aggregation gives a silently incorrect count.
* **Inserting rows without listing the column names**: `INSERT INTO t VALUES (1, 'foo')` breaks silently or inserts wrong data the moment a column is added, reordered, or has a different default. Always list column names explicitly in every INSERT statement.

## Next steps

* [Joins](joins.md), combine rows from multiple tables (next in the lesson sequence)
* [Aggregations](aggregations.md), **GROUP BY**, aggregate functions, **HAVING**
* [Advanced SQL concepts](advanced-concepts.md), window functions, CTEs, and optimization
