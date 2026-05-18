# 2.1 SQL — Hints

For each task we point at the lesson section and give a **way to think about the query** plus a starter pattern. We never give the full query.

> Open [the assignment](./coding.md) in another tab. Run queries against the `chinook` database in DBeaver.

## 1. Basic queries and filtering
- **Where:** [Basic operations](../basic-operations.md).
- **Think:**
  - **Tracks > $0.99:** `SELECT … WHERE UnitPrice > 0.99`. Find which table holds prices first (browse the schema in DBeaver).
  - **"Love" in name:** `LIKE '%Love%'`. Case sensitivity depends on the DB collation — SQLite is case-insensitive for ASCII `LIKE` by default.
  - **Sales Agents:** filter on the title column in `employees`. Look up exact wording — it may be "Sales Support Agent" rather than the prompt's "Sales Agent".

## 2. Advanced filtering and sorting
- **Where:** [Basic operations](../basic-operations.md) — `WHERE`, `BETWEEN`, `ORDER BY`.
- **Think:**
  - **5–10 min tracks:** the column is `Milliseconds`. 5 min = 300,000 ms, 10 min = 600,000 ms. `BETWEEN 300000 AND 600000`.
  - **First-name starts with 'A', sorted by country:** `WHERE FirstName LIKE 'A%' ORDER BY Country`.
  - **Invoices from 2009:** date column on `invoices` — filter with `strftime('%Y', InvoiceDate) = '2009'` (SQLite) or `EXTRACT(YEAR FROM …)` (Postgres). Sort by `Total DESC`.

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
  - **Tracks with album / genre / media type:** four-way join — start from `tracks` and join out to each lookup table.
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
- Joining on the wrong foreign key — read the schema diagram first.
- SQLite stores dates as text; comparisons need `strftime` or string-prefix tricks.
