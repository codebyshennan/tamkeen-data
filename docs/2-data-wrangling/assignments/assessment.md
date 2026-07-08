# Module 2 Assessment

> **Submit your work on Skills Union →** [https://skillsu.com/member/assessment](https://skillsu.com/member/assessment)

**Mandatory** • 10 questions • Covers SQL, Data Wrangling, EDA, and Data Engineering.

Try each question closed-book first. Click **Show hint** if you get stuck, hints point you at the relevant lesson section and how to think about it, without naming the answer.

***

**Q1.** What SQL clause specifies which table(s) to query?

* SELECT
* FROM
* WHERE
* ORDER BY

<details>

<summary>Show hint</summary>

* **Where:** [SQL basics](../2.1-sql/), anatomy of a `SELECT` statement.
* **Think:** Walk through what each clause does. One picks columns, one filters rows, one sorts. The remaining clause names the **source** the query reads from.

</details>

**Q2.** What SQL keyword is used to make table aliases more readable, especially with multiple joins?

* SELECT
* INSERT
* LIKE
* AS

<details>

<summary>Show hint</summary>

* **Where:** [SQL basics](../2.1-sql/), aliases and joins.
* **Think:** Aliases rename something for the rest of the query (e.g. `customers c`). Which keyword reads naturally as "this table, **named as** something shorter"?

</details>

**Q3.** What pandas DataFrame method returns a Series containing column sums?

* `mean()`
* `median()`
* `sum()`
* `mode()`

<details>

<summary>Show hint</summary>

* **Where:** [Data wrangling with pandas](../2.2-data-wrangling/), reductions.
* **Think:** Each option does what its name says. Pick the one whose name literally describes "adding values together".

</details>

**Q4.** What method on a Series accepts a function or dict-like mapping to transform values?

* `apply`
* `groupby`
* `filter`
* `map`

<details>

<summary>Show hint</summary>

* **Where:** [Data wrangling with pandas](../2.2-data-wrangling/), element-wise transforms.
* **Think:** `groupby` and `filter` don't take a dict. Of the two remaining, one is specifically the **Series**-level transform that accepts either a function **or** a dict, the other lives on DataFrames and on Series but is more general.

</details>

**Q5.** Permuting (randomly reordering) a Series or the rows in a DataFrame is possible using what NumPy function?

* `numpy.transpose`
* `numpy.random.shuffle`
* `numpy.permute`
* `numpy.random.permutation`

<details>

<summary>Show hint</summary>

* **Where:** [Data wrangling with pandas](../2.2-data-wrangling/), sampling and shuffling.
* **Think:** `transpose` swaps axes (not a shuffle). One option doesn't exist in NumPy. Of the two real shuffles, one mutates **in place** and returns nothing, the other **returns** a new permuted array, which is what you want for indexing rows.

</details>

**Q6.** What is a measure of how much two random variables vary together?

* Standard deviation
* Mean
* Variance
* Covariance

<details>

<summary>Show hint</summary>

* **Where:** [EDA](../2.3-eda/), relationships between variables.
* **Think:** Three options describe a **single** variable's centre or spread. Only one captures the **joint** behaviour of two variables, and its name starts with the prefix meaning "together".

</details>

**Q7.** The correlation coefficient indicates the strength of a linear relationship between variables. What range can it take?

* -2 to 2
* 0 to 1
* -1 to 1
* 0 to infinity

<details>

<summary>Show hint</summary>

* **Where:** [EDA](../2.3-eda/), Pearson correlation.
* **Think:** Correlation captures **direction** (positive or negative) and **strength** (how close to perfect). So the range must be **symmetric around zero** and **bounded on both sides**.

</details>

**Q8.** What pandas method connects rows in DataFrames based on one or more keys?

* `pivot()`
* `groupby()`
* `merge()`
* `transform()`

<details>

<summary>Show hint</summary>

* **Where:** [Data wrangling with pandas](../2.2-data-wrangling/), combining DataFrames.
* **Think:** Three options reshape or aggregate a **single** DataFrame. Only one joins **two** DataFrames together on key columns, its name mirrors the SQL operation that does the same thing.

</details>

**Q9.** Which is _not_ a component of a data pipeline?

* Data storage
* Data processing
* Data ingestion
* Data release

<details>

<summary>Show hint</summary>

* **Where:** [Data engineering overview](../2.2-data-wrangling/), pipeline stages.
* **Think:** A pipeline takes data **in**, does something **to** it, and puts it **somewhere**. One of these options uses a verb that isn't part of that standard vocabulary, pipelines don't "release" data, they store / serve / publish it.

</details>

**Q10.** What does the ETL pattern stand for?

* Edit, Transfer, Load
* Evaluate, Transform, Link
* Encrypt, Transmit, Log
* Extract, Transform, Load

<details>

<summary>Show hint</summary>

* **Where:** [Data engineering](../2.2-data-wrangling/), ETL.
* **Think:** Read each option's letters. The middle word is the same in two options, that's a strong clue it's correct. Between those two, which **first** word matches "pull data out of a source"?

</details>
