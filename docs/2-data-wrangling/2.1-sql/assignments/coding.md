# Assignment: Data Querying with SQL

> **Submit your work on Skills Union →** <https://skillsu.com/member/assessment>

## Setup

1. Connect to the `chinook` database using DBeaver as explained in the lessons

## Tasks

1. Basic Queries and Filtering

   - List all tracks that cost more than $0.99
   - List all tracks with the word "Love" in their name
   - Show all employees who are Sales Agents

2. Advanced Filtering and Sorting

   - List all tracks that are between 5 and 10 minutes long (use Milliseconds)
   - Find all customers whose first name starts with 'A' and sort them by country
   - List all invoices from 2009, sorted by total amount (highest to lowest)

3. Aggregation and Grouping

   - Find the total number of tracks in each album
   - Show the total sales amount for each country

4. Multiple Table Operations
   - Show all albums with their artist names
   - List all tracks with their album name, genre, and media type
   - Show the total number of tracks purchased by each customer

## Deliverable

Submit your solution as a SQL script (`.sql`) with:

1. All queries clearly commented with
   1. Brief explanations of your approach for complex operations
   2. Any assumptions or additional features you implemented

## Hints

<details>
<summary>Show hints</summary>

## 1. Basic queries and filtering
- **Where:** [Basic operations](../basic-operations.md).
- **Think:**
  - **Tracks > $0.99:** `SELECT … WHERE UnitPrice > 0.99`. Find which table holds prices first (browse the schema in DBeaver).
  - **"Love" in name:** `LIKE '%Love%'`. Case sensitivity depends on the DB collation, SQLite is case-insensitive for ASCII `LIKE` by default.
  - **Sales Agents:** filter on the title column in `employees`. Look up exact wording, it may be "Sales Support Agent" rather than the prompt's "Sales Agent".

## 2. Advanced filtering and sorting
- **Where:** [Basic operations](../basic-operations.md), `WHERE`, `BETWEEN`, `ORDER BY`.
- **Think:**
  - **5-10 min tracks:** the column is `Milliseconds`. 5 min = 300,000 ms, 10 min = 600,000 ms. `BETWEEN 300000 AND 600000`.
  - **First-name starts with 'A', sorted by country:** `WHERE FirstName LIKE 'A%' ORDER BY Country`.
  - **Invoices from 2009:** date column on `invoices`, filter with `strftime('%Y', InvoiceDate) = '2009'` (SQLite) or `EXTRACT(YEAR FROM …)` (Postgres). Sort by `Total DESC`.

## 3. Aggregation and grouping
- **Where:** [Aggregations](../aggregations.md).
- **Think:**
  - **Tracks per album:** `GROUP BY AlbumId`, `COUNT(*)`. Optionally join `albums` to show the title.
  - **Sales per country:** sum from invoices grouped by `BillingCountry`.
- **Starter:**
  ```sql
  SELECT AlbumId, COUNT(*) AS track_count
  FROM tracks
  GROUP BY AlbumId
  ORDER BY track_count DESC;
  ```

## 4. Multiple table operations
- **Where:** [Joins](../joins.md).
- **Think:**
  - **Albums with artist names:** classic two-table join on `ArtistId`.
  - **Tracks with album / genre / media type:** four-way join, start from `tracks` and join out to each lookup table.
  - **Tracks purchased per customer:** chain `customers → invoices → invoice_items`, then `SUM(Quantity)` grouped by customer.
- **Starter:**
  ```sql
  SELECT al.Title, ar.Name AS artist
  FROM albums al
  JOIN artists ar ON ar.ArtistId = al.ArtistId;
  ```

## Common pitfalls
- Confusing `WHERE` (per row) with `HAVING` (per group).
- Forgetting `GROUP BY` for every non-aggregated select column.
- Joining on the wrong foreign key, read the schema diagram first.
- SQLite stores dates as text; comparisons need `strftime` or string-prefix tricks.

</details>
