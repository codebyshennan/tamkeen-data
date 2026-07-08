# Mastering SQL Joins: Connecting Your Data Universe

**After this lesson:** You can choose the right join type (**INNER**, **LEFT**, **RIGHT**, **FULL**) for a question, write multi-table queries with clear aliases, and avoid accidental Cartesian products.

## Helpful video

Quick tour of join types in SQL (inner, left, right, full).

## Overview

**Prerequisites:** [Basic SQL Operations](basic-operations.md) and [Aggregations](aggregations.md). You should recognize foreign keys from [Introduction to Databases](intro-databases.md).

> **Time needed:** About 60-90 minutes with practice queries.

## Why this matters

Almost every real question spans more than one table, customers and orders, students and enrollments, parts and suppliers. Choosing **INNER** vs **LEFT** join is choosing _which rows you are willing to drop_ from the result; getting that wrong silently loses data or duplicates it.

## Introduction to SQL Joins

> **Figure (add screenshot or diagram):** Four Venn diagrams side by side, INNER (centre only), LEFT (left circle + centre), RIGHT (right circle + centre), FULL OUTER (both circles). Shade the returned region for each.

SQL joins combine rows from two or more tables based on related columns. They are essential for:

* Retrieving related data across tables
* Building comprehensive reports
* Analyzing relationships in data
* Creating meaningful insights

## Types of SQL Joins

### 1. INNER JOIN

Returns only matching rows from both tables.

Basic INNER JOIN

Joins `orders` to `customers` on `customer_id`. Only rows with a matching customer in both tables appear, orders with no customer record and customers with no orders are both excluded.

Multiple ON conditions

Two predicates in the `ON` clause mean both must match: the order's customer and the store must be the customer's preferred store. This further restricts the result, only orders placed at the customer's preferred location are returned.

### 2. LEFT JOIN (LEFT OUTER JOIN)

Returns all rows from the left table and matching rows from the right table.

LEFT JOIN: all customers, even those with no orders

Every customer from the left table appears in the result. Where there are no matching orders, `COUNT(o.order_id)` returns 0 and `COALESCE(SUM(…), 0)` substitutes 0 for the NULL total, customers who never ordered still appear with zeroes.

Finding customers with no orders

After the LEFT JOIN, rows where `o.order_id IS NULL` are exactly the customers who have no matching order. This anti-join pattern is a reliable way to find "missing" relationships without subqueries.

### 3. RIGHT JOIN (RIGHT OUTER JOIN)

Returns all rows from the right table and matching rows from the left table.

RIGHT JOIN: all products, even those never ordered

The right table (`products`) drives the result, every product appears regardless of whether it has any matching `order_items` rows. `COALESCE(SUM(oi.quantity), 0)` returns 0 for products with no orders instead of NULL.

Finding products that have never been ordered

After the RIGHT JOIN, `WHERE oi.order_id IS NULL` isolates products with no matching order item, another anti-join pattern, this time preserving the right table's unmatched rows.

### 4. FULL JOIN (FULL OUTER JOIN)

Returns all rows when there's a match in either left or right table.

FULL JOIN: every row from every table

Three chained FULL JOINs mean unmatched rows from any table still appear, a customer with no orders, an order with no items, and a product with no order items all show up with NULLs for the unjoined columns.

Finding all missing relationships

`COALESCE` substitutes readable labels for NULLs in the output. `WHERE o.order_id IS NULL` filters to only the orphaned rows, customers with no orders, or products never ordered, exposing data integrity gaps across the four tables.

### 5. CROSS JOIN

Returns Cartesian product of both tables.

Basic CROSS JOIN: every product × category combination

A CROSS JOIN has no `ON` condition, it produces every combination of rows from both tables. With 100 products and 10 categories this gives 1,000 rows. Useful for generating all possibilities (e.g., a pricing matrix), dangerous when accidental.

Generate date × product combinations

`generate_series` produces 8 consecutive dates; CROSS JOIN pairs each date with every product. This is a common pattern for pre-filling a calendar grid so that days with zero sales still appear as rows rather than gaps.

## Common Join Patterns

### 1. Multi-Table Joins

Multi-table INNER JOIN chain

Three consecutive JOINs thread through four tables: `orders → customers` for the buyer name, `orders → order_items` for the line rows, `order_items → products` for the price. Each JOIN adds columns; only rows present in every table are kept.

### 2. Self Joins

Self join: employee → manager from the same table

The same `employees` table is aliased twice. `e` is the employee row; `m` is the manager row found by matching `e.manager_id = m.employee_id`. LEFT JOIN keeps employees who have no manager (e.g., the CEO).

Self join on order\_items: frequently bought together

`order_items` is joined to itself on the same `order_id` to find pairs of products appearing in the same order. `oi1.product_id < oi2.product_id` prevents counting each pair twice and eliminates self-pairs. `HAVING COUNT(*) > 5` keeps only popular combinations.

### 3. Conditional Joins

Range join: match events to overlapping promotions

The `ON` clause uses `BETWEEN` instead of equality, an event matches a promotion if its date falls within the promotion's active window. LEFT JOIN keeps events even when no promotion was running.

Multi-condition join: only active drivers with capacity

Three conditions in the `ON` clause act as a composite filter at join time: the driver must cover the order's delivery zone, be currently active, and have fewer orders than their maximum capacity. This keeps the `WHERE` clause clean and expresses driver eligibility as part of the join.

## Join Best Practices

### 1. Performance Optimization

Index join columns for fast lookups

Creating indexes on the foreign key columns used in `ON` clauses (`customer_id`, and the composite `order_id, product_id`) lets the planner use an index nested-loop join instead of a full sequential scan of each table.

Hint: start from the smallest table

The `/*+ LEADING(…) */` hint tells the planner which table to access first. Joining small → medium → large reduces intermediate row counts at each step and keeps memory usage low.

### 2. Common Mistakes to Avoid

Avoid implicit Cartesian products

The comma-separated `FROM orders, customers` syntax produces a full Cartesian product, every order row paired with every customer row. Without a filter this is almost always a mistake. The explicit `JOIN … ON` form makes the intent clear and is harder to accidentally omit.

Handle NULLs from outer joins with COALESCE

After a LEFT JOIN, unmatched right-table columns are NULL. Aggregating NULL values with `SUM` returns NULL, not 0. `COALESCE(SUM(o.total_amount), 0)` converts the NULL result to 0 so customers who have never ordered show a meaningful total.

### 3. Maintainability Tips

Readable aliases: full words instead of single letters

Using `cust`, `ord`, and `prod` instead of `c`, `o`, `p` makes queries self-documenting, readers can tell which table each column comes from without cross-referencing the FROM clause.

CTE 1: per-customer order counts

The first CTE pre-aggregates orders into one row per customer. Breaking a complex join into CTEs makes each piece independently readable and testable before composing them in the final SELECT.

CTE 2 + outer join: combine aggregates per customer

The second CTE sums total spend per customer. The final SELECT LEFT JOINs both CTEs onto `customers` so that customers with no orders still appear, with NULLs for count and total rather than being silently dropped.

## Additional Real-World Scenarios

### 1. E-commerce Funnel Analysis

CTE: per-user event counts across the funnel

LEFT JOIN `events` keeps users who never triggered any event. `COUNT(DISTINCT CASE WHEN event_type = 'view' THEN product_id END)` counts unique products at each funnel stage, view, cart, purchase, per user without multiple self-joins.

Outer query: aggregate funnel conversion rates

Averages per-user counts across all users, then computes view-to-cart and cart-to-purchase rates as percentages. `NULLIF(…, 0)` in the denominator prevents division-by-zero when no users reached the prior stage.

### 2. Supply Chain Analysis

CTE: join suppliers → orders → deliveries

Two LEFT JOINs chain from supplier to purchase orders then to deliveries. LEFT JOIN keeps suppliers who have no orders or deliveries. `EXTRACT(EPOCH FROM …)/86400` converts the interval to fractional days for `AVG`.

Outer query: compute late-delivery rate and rating

Divides late deliveries by total orders-`NULLIF(orders_fulfilled, 0)` prevents division-by-zero for new suppliers. The `CASE` expression buckets each supplier into a performance tier (Excellent / Good / Fair / Poor) for easy reporting.

### 3. Customer Service Integration

CTE: join support tickets → orders → products

Three LEFT JOINs link each ticket to its related order, the order's items, and those items' products. LEFT JOIN preserves tickets that aren't linked to an order. `EXTRACT(EPOCH FROM …)/3600` converts the timestamp difference to hours for resolution time.

Outer query: aggregate metrics by priority

Groups tickets by priority and calculates ticket count, average resolution time, resolution rate, and a comma-separated list of distinct affected products. `FILTER (WHERE product_name IS NOT NULL)` on `STRING_AGG` skips tickets not linked to any product.

## Performance Optimization Examples

### 1. Hash Join vs. Merge Join

Hash join: best for large unsorted tables

Hash joins build an in-memory hash table from the smaller table, then probe it for each row of the larger table. They work well when neither side has a useful index and are typically the planner's default for large ad-hoc joins.

Merge join: best for pre-sorted or indexed columns

Merge joins require both sides sorted on the join key. When an index already provides that order the planner can avoid a sort step and stream through both sides in a single pass, very efficient for equality joins on indexed columns.

### 2. Partitioned Joins

Declare partitioned parent tables

`PARTITION BY RANGE (order_date)` and `PARTITION BY RANGE (order_id)` create parent tables with no data of their own, they delegate rows to child partitions. Queries against the parent automatically target only the relevant partition(s).

Create child partitions and join them directly

Each `PARTITION OF … FOR VALUES FROM … TO …` creates a child table holding rows in that range. Joining the Q1 child partitions directly instead of the parent tables lets the planner skip all other partitions entirely, partition pruning.

### 3. Materialized Views for Complex Joins
