# Module 1 Assignment — Hints

**Audience:** Students who want a nudge, not the answer. For each question we point you at the **lesson section** that covers the concept and give a **way to think about it**. The answer key is a separate file for instructors.

> Open the [Module 1 assignment](./module-assignment.md) in another tab and use these hints when you are stuck.

> **Hint convention.** A hint should make a confused student **less confused without telling them the answer**. If your hint contains the exact keyword from the correct option, rephrase it.

---

## Part 1 — Introduction to Data Analytics

Lesson root: [1.1 Introduction to data analytics](../1.1-intro-data-analytics/README.md).

### Q1 — purpose of data collection

- **Where to look:** [Data Collection](../1.1-intro-data-analytics/data-collection.md) — opening section.
- **How to think:** Three of the four options describe what you do **after** data exists. Only one describes the step that **produces** data in the first place.

### Q2 — first-party data example

- **Where to look:** [Data Collection](../1.1-intro-data-analytics/data-collection.md) — "First/second/third-party data".
- **How to think:** Ask **"who collected it, and from whom?"** First-party = the data you collected yourself from people who interacted with you. Reports you bought are someone else's collection.

### Q3 — what GDPR stands for

- **Where to look:** [Data Privacy](../1.1-intro-data-analytics/data-privacy.md).
- **How to think:** This question intentionally has two options that look almost identical — read each letter carefully. The word "Privacy" vs "Protection" is the key disambiguator, and "General" vs "Global" matters too.

### Q4 — gathering qualitative data

- **Where to look:** [Data Collection](../1.1-intro-data-analytics/data-collection.md) — "Quantitative vs qualitative methods".
- **How to think:** Qualitative = words, meaning, context (not numbers). Which option produces a transcript or a story, not a table of measurements?

### Q5 — focus of predictive analytics

- **Where to look:** [Workflow Concepts](../1.1-intro-data-analytics/workflow-concepts.md) — descriptive / diagnostic / predictive / prescriptive ladder.
- **How to think:** The four analytics types map to **tenses**: past (descriptive), why-past (diagnostic), **future (predictive)**, what-to-do (prescriptive). Pick the option whose verb is future tense.

### Q6 — what is data cleaning

- **Where to look:** [Workflow Concepts](../1.1-intro-data-analytics/workflow-concepts.md) — the "Process" section.
- **How to think:** Several listed activities (dedupe, error correction, reformatting) are all real cleaning tasks. If more than one option is individually true, look for an option that names the whole umbrella.

### Q7 — direct observation data type

- **Where to look:** [Data Collection](../1.1-intro-data-analytics/data-collection.md) — primary vs secondary.
- **How to think:** Two of the options ("tertiary", "meta") are distractors that aren't standard terms in this taxonomy. Of the remaining two, which one means **you observed it yourself** vs. **someone else handed it to you**?

### Q8 — purpose of data visualization

- **Where to look:** [Workflow Concepts](../1.1-intro-data-analytics/workflow-concepts.md) — communication and visualization.
- **How to think:** "Visualization" literally means making something visible. Three options describe other workflow steps; one matches that definition.

### Q9 — characteristic of big data

- **Where to look:** [Data Collection](../1.1-intro-data-analytics/data-collection.md) — "Big data" / 3 Vs.
- **How to think:** The famous "Vs" of big data each name a real characteristic. If the options list them all individually plus an "all of the above", the umbrella option is usually the intended pick.

### Q10 — what is data mining

- **Where to look:** [Workflow Concepts](../1.1-intro-data-analytics/workflow-concepts.md) — analysis methods.
- **How to think:** "Mining" implies digging up something hidden. Which option describes **discovering** something rather than just storing or moving it?

### Q11 — not a type of data analytics

- **Where to look:** [Workflow Concepts](../1.1-intro-data-analytics/workflow-concepts.md) — analytics types ladder.
- **How to think:** Recall the four official types (descriptive / diagnostic / predictive / prescriptive). The odd one out is the option that sounds plausible but is **not** on that list.

### Q12 — role of a data analyst

- **Where to look:** [README](../1.1-intro-data-analytics/README.md) — "What is a data analyst?".
- **How to think:** Three options use the word "only" — that's the giveaway. A real analyst does more than any single one of those tasks.

### Q13 — example of structured data

- **Where to look:** [Data Collection](../1.1-intro-data-analytics/data-collection.md) — structured vs unstructured.
- **How to think:** Structured = fits neatly into rows and columns with a schema. Three of the options are free-form content; one is a tabular store.

### Q14 — what is data governance

- **Where to look:** [Data Security](../1.1-intro-data-analytics/data-security.md) and [Data Privacy](../1.1-intro-data-analytics/data-privacy.md).
- **How to think:** "Governance" is about **rules and oversight**, not about doing the work itself. Pick the option that sounds like a policy framework, not a specific activity.

### Q15 — factor for data quality

- **Where to look:** [Data Collection](../1.1-intro-data-analytics/data-collection.md) — data quality dimensions.
- **How to think:** Volume, age, and source can all be high without the data being **correct**. Which option directly names correctness?

### Q16 — what is metadata

- **Where to look:** [Data Collection](../1.1-intro-data-analytics/data-collection.md) — metadata sub-section.
- **How to think:** The prefix "meta-" means "about" (meta-analysis = analysis about analysis). Apply that to the word "data".

### Q17 — not a common collection method

- **Where to look:** [Data Collection](../1.1-intro-data-analytics/data-collection.md).
- **How to think:** Three options are real techniques you'd find in any research methods text. One is a joke distractor — pick the one that does not exist as a real method.

### Q18 — purpose of data aggregation

- **Where to look:** [Workflow Concepts](../1.1-intro-data-analytics/workflow-concepts.md) — aggregation step.
- **How to think:** The word "aggregate" means **to gather together into one**. Which option literally describes combining inputs?

### Q19 — benefit of real-time analytics

- **Where to look:** [Workflow Concepts](../1.1-intro-data-analytics/workflow-concepts.md) — real-time / batch.
- **How to think:** The whole point of "real-time" is **speed of feedback**. Lower cost and simpler analysis are usually trade-offs, not benefits, of real-time pipelines.

### Q20 — what is data integrity

- **Where to look:** [Data Security](../1.1-intro-data-analytics/data-security.md) — integrity sub-section.
- **How to think:** "Integrity" in everyday English = wholeness, trustworthiness. Two options are about confidentiality (encryption) or availability (backup, compression); only one is about whether the data itself is **right and unchanged**.

---

## Part 2 — Python Programming

Lesson root: [1.2 Introduction to Python](../1.2-intro-python/README.md).

### Q1 — declaring a list

- **Where to look:** [Data Structures](../1.2-intro-python/data-structures.md) — list literals.
- **How to think:** Each bracket style in Python builds a different container — `()` tuple, `{}` set/dict, `<>` not a Python literal at all. Which bracket pair builds an ordered, mutable sequence?

### Q2 — `len(['a', 'b', ['c', 'd']])`

- **Where to look:** [Data Structures](../1.2-intro-python/data-structures.md) — nested lists.
- **How to think:** `len()` only counts the **top-level** items. A nested list counts as **one** item regardless of how many things are inside it. Read the outer brackets and count.

### Q3 — creating a dictionary

- **Where to look:** [Data Structures](../1.2-intro-python/data-structures.md) — dict literals.
- **How to think:** Dicts use the same brackets as sets but with `key: value` pairs inside. Three of the options use wrong bracket styles entirely.

### Q4 — `3 * 'abc'`

- **Where to look:** [Basic Syntax / Data Types](../1.2-intro-python/basic-syntax-data-types.md) — string operators.
- **How to think:** In Python, `*` between an int and a string means **repeat**. So `3 * 'abc'` gives three copies concatenated together.

### Q5 — adding an item to a list

- **Where to look:** [Data Structures](../1.2-intro-python/data-structures.md) — list methods table.
- **How to think:** Several real list methods can add items, but only one is the **single-item add to the end**. `add()` is for sets; `extend()` takes an iterable; `insert()` needs an index.

### Q6 — `'Hello' + 'World'`

- **Where to look:** [Basic Syntax / Data Types](../1.2-intro-python/basic-syntax-data-types.md) — string concatenation.
- **How to think:** `+` between strings just sticks them together character-for-character. No space appears unless one of the strings contains one.

### Q7 — exponentiation operator

- **Where to look:** [Basic Syntax / Data Types](../1.2-intro-python/basic-syntax-data-types.md) — arithmetic operators.
- **How to think:** `^` is bitwise XOR in Python, not power. `//` is floor division. The "power" operator doubles a familiar symbol.

### Q8 — `bool(0)`

- **Where to look:** [Basic Syntax / Data Types](../1.2-intro-python/basic-syntax-data-types.md) — truthiness.
- **How to think:** `bool(x)` returns a **boolean**, not a number. In Python's truthiness rules, zero counts as the "empty/none" side.

### Q9 — string to lowercase

- **Where to look:** [Basic Syntax / Data Types](../1.2-intro-python/basic-syntax-data-types.md) — string methods.
- **How to think:** Python style is `verb()` not `toVerb()` (that's JS). Method, not function. Which option is shortest and most Pythonic?

### Q10 — what `split()` does

- **Where to look:** [Basic Syntax / Data Types](../1.2-intro-python/basic-syntax-data-types.md) — string methods.
- **How to think:** `split` is called on a **string** and produces a **list of substrings**. Reverse operation of `join`.

### Q11 — strip whitespace from both ends

- **Where to look:** [Basic Syntax / Data Types](../1.2-intro-python/basic-syntax-data-types.md) — string methods.
- **How to think:** Two options are not real Python methods (one is from JavaScript). Of the two real ones, which removes whitespace specifically and from both ends?

### Q12 — creating a function

- **Where to look:** [Functions](../1.2-intro-python/functions.md) — function definition syntax.
- **How to think:** Python's function-defining keyword is short (3 letters) and not borrowed from JS, C, or PHP. You see it on the first line of every Python function you have read.

### Q13 — length of a tuple

- **Where to look:** [Data Structures](../1.2-intro-python/data-structures.md) — tuples.
- **How to think:** Python uses a **built-in function**, not a method, for length — the same one that works on lists, strings, dicts, and tuples.

### Q14 — importing a module

- **Where to look:** [Modules](../1.2-intro-python/modules.md).
- **How to think:** Three of the options come from other languages (C, C#, Ruby). Pick the one-word Python keyword you have already typed at the top of every notebook.

### Q15 — add to end of list

- **Where to look:** [Data Structures](../1.2-intro-python/data-structures.md) — list methods.
- **How to think:** Same answer as Q5 — Python's "push to end" method is **not** called `push` (that's JS) or `add` (that's set). It is a six-letter verb meaning "attach to the end".

### Q16 — `type([1, 2, 3])`

- **Where to look:** [Basic Syntax / Data Types](../1.2-intro-python/basic-syntax-data-types.md) — type names.
- **How to think:** `type(x)` returns the class of `x`. `[1, 2, 3]` is built with square brackets, which makes it a... (Python does not have a built-in `array` type at the top level.)

### Q17 — empty dictionary

- **Where to look:** [Data Structures](../1.2-intro-python/data-structures.md) — dict literals and constructor.
- **How to think:** Both the literal `{}` and the constructor `dict()` give you an empty dict. Watch for an option that says "both".

### Q18 — `10 / 3`

- **Where to look:** [Basic Syntax / Data Types](../1.2-intro-python/basic-syntax-data-types.md) — true division vs floor division.
- **How to think:** In Python 3, `/` is **true division** and always returns a `float`. Do not confuse it with `//` (floor) or with how Python 2 worked.

### Q19 — invalid variable name

- **Where to look:** [Basic Syntax / Data Types](../1.2-intro-python/basic-syntax-data-types.md) — identifier rules.
- **How to think:** Python identifiers can contain letters, digits, and underscores, but **cannot start with a digit**. Scan each option's first character.

### Q20 — `'python'[1:4]`

- **Where to look:** [Basic Syntax / Data Types](../1.2-intro-python/basic-syntax-data-types.md) — string slicing.
- **How to think:** Slice `[a:b]` takes characters from index `a` up to **but not including** index `b`. Index 0 is `'p'`. Now count.

---

## Part 3 — Statistics

Lesson root: [1.3 Introduction to Statistics](../1.3-intro-statistics/README.md).

### Q1 — mean of 2, 4, 6, 8, 10

- **Where to look:** [One-variable statistics](../1.3-intro-statistics/one-variable-statistics.md) — mean.
- **How to think:** Sum the values, divide by the count. Five values, evenly spaced around their middle — the answer should equal the middle value.

### Q2 — central tendency most affected by outliers

- **Where to look:** [One-variable statistics](../1.3-intro-statistics/one-variable-statistics.md) — mean vs median vs mode.
- **How to think:** Range is a spread measure, not central tendency. Of the remaining three, only one is computed by **summing every value** — that is the one a single huge number distorts.

### Q3 — P(rolling a 6 on a fair die)

- **Where to look:** [Probability fundamentals](../1.3-intro-statistics/probability-fundamentals.md) — equally likely outcomes.
- **How to think:** Fair die = each face equally likely. One favourable face out of six total.

### Q4 — symmetric bell-shaped distribution

- **Where to look:** [Probability distribution families](../1.3-intro-statistics/probability-distribution-families.md).
- **How to think:** Uniform is flat. Exponential is heavily skewed. Binomial is discrete. Which family produced the most famous bell curve in statistics?

### Q5 — what standard deviation measures

- **Where to look:** [One-variable statistics](../1.3-intro-statistics/one-variable-statistics.md) — variance and SD.
- **How to think:** SD answers "how spread out are the values around the mean?" — that is the definition of a **spread** (variability) measure, not central tendency or position.

### Q6 — median of 1, 3, 3, 6, 7, 8, 9

- **Where to look:** [One-variable statistics](../1.3-intro-statistics/one-variable-statistics.md) — median.
- **How to think:** The list is already sorted. There are seven values, so the median is the **middle one** (4th position). Count: 1, 3, 3, **?**, 7, 8, 9.

### Q7 — correlation: as one ↑, the other ↓

- **Where to look:** [Two-variable statistics](../1.3-intro-statistics/two-variable-statistics.md) — correlation.
- **How to think:** The sign of a correlation tells you direction. Positive = same direction. Zero = no linear relationship. "Perfect" describes the strength, not direction.

### Q8 — range of correlation coefficient

- **Where to look:** [Two-variable statistics](../1.3-intro-statistics/two-variable-statistics.md) — Pearson r.
- **How to think:** Pearson's *r* is symmetric: perfect negative is `-1`, no correlation is `0`, perfect positive is `+1`. The full range is bounded on both ends.

### Q9 — which measure resists outliers

- **Where to look:** [One-variable statistics](../1.3-intro-statistics/one-variable-statistics.md) — robust statistics.
- **How to think:** The mean, range, and variance all use every value (so one big outlier shifts them a lot). The one that depends only on **rank order** does not care if the biggest number is 100 or 10,000.

### Q10 — mode of 2, 2, 3, 4, 4, 4, 5

- **Where to look:** [One-variable statistics](../1.3-intro-statistics/one-variable-statistics.md) — mode.
- **How to think:** Mode = most frequent value. Count how often each number appears.

### Q11 — P(heads on a fair coin)

- **Where to look:** [Probability fundamentals](../1.3-intro-statistics/probability-fundamentals.md).
- **How to think:** Fair coin = two equally likely outcomes. One favourable, two total.

### Q12 — measure of spread that is always positive

- **Where to look:** [One-variable statistics](../1.3-intro-statistics/one-variable-statistics.md) — variance.
- **How to think:** Mean and median can be negative (e.g. negative numbers in the data). Correlation can be negative too. The remaining option is defined as a sum of **squared** deviations — squares are never negative.

### Q13 — interquartile range

- **Where to look:** [One-variable statistics](../1.3-intro-statistics/one-variable-statistics.md) — quartiles & IQR.
- **How to think:** "Inter-quartile" literally means "between the quartiles". Q1 is the 25th percentile, Q3 is the 75th. IQR is the gap between them.

### Q14 — type of variable for "age"

- **Where to look:** [One-variable statistics](../1.3-intro-statistics/one-variable-statistics.md) — nominal / ordinal / interval / ratio.
- **How to think:** Age has a **true zero** (zero years = no age elapsed) and ratios are meaningful (a 40-year-old is twice as old as a 20-year-old). That is the strongest scale of measurement.

### Q15 — what is a percentile

- **Where to look:** [One-variable statistics](../1.3-intro-statistics/one-variable-statistics.md) — percentiles.
- **How to think:** A percentile tells you where a value sits **relative** to the rest of the distribution. It is not a single summary of the centre or the spread; it locates a point.

### Q16 — distribution for count data

- **Where to look:** [Probability distribution families](../1.3-intro-statistics/probability-distribution-families.md) — Poisson.
- **How to think:** Normal is for continuous symmetric data. Uniform is flat over a range. Exponential models waiting times. The remaining one is the discrete distribution for "how many events happened in an interval".

### Q17 — law of large numbers

- **Where to look:** [Probability fundamentals](../1.3-intro-statistics/probability-fundamentals.md).
- **How to think:** LLN is about what happens to the **sample mean** as you collect more and more samples — it stabilises around the true average. Find the option that says exactly that.

### Q18 — test to compare two population means

- **Where to look:** [Probability distributions](../1.3-intro-statistics/probability-distributions.md) — common tests.
- **How to think:** Chi-square is for categorical data. F-test compares variances. z-test needs known population variance and is less common in practice. The remaining one is the workhorse for "are these two group means different?".

### Q19 — what is a Type I error

- **Where to look:** [Probability distributions](../1.3-intro-statistics/probability-distributions.md) — hypothesis errors.
- **How to think:** Type I = "false alarm" = you flagged a difference that was not really there. In null-hypothesis language: you **rejected** something that was actually **true**.

### Q20 — purpose of a confidence interval

- **Where to look:** [Probability distributions](../1.3-intro-statistics/probability-distributions.md) — confidence intervals.
- **How to think:** A CI gives you a **range of plausible values** for a population parameter (like the mean). It is an estimate-with-uncertainty, not a test or a correlation.

---

## Part 4 — NumPy Operations

Lesson root: [1.4 Data foundation: linear algebra & NumPy](../1.4-data-foundation-linear-algebra/README.md).

### Q1 — create an array of zeros

- **Where to look:** [Intro to NumPy](../1.4-data-foundation-linear-algebra/intro-numpy.md) — array constructors.
- **How to think:** NumPy names its constructors after **what they fill the array with**. There is a `np.ones()` too. The "blank" / "null" options do not exist as functions.

### Q2 — shape of an array

- **Where to look:** [ndarray basics](../1.4-data-foundation-linear-algebra/ndarray-basic.md) — attributes.
- **How to think:** `shape` is an **attribute** (no parentheses), not a method. The other options use names that are not part of the ndarray API.

### Q3 — matrix multiplication

- **Where to look:** [Linear algebra](../1.4-data-foundation-linear-algebra/linear-algebra.md).
- **How to think:** `np.multiply()` is element-wise (Hadamard product), not matrix multiplication. Both `np.matmul()` and `np.dot()` perform matrix multiplication for 2-D arrays. Look for an option that acknowledges both.

### Q4 — evenly spaced numbers

- **Where to look:** [ndarray basics](../1.4-data-foundation-linear-algebra/ndarray-basic.md) — `linspace` vs `arange`.
- **How to think:** Both `linspace` and `arange` create evenly spaced values, but with different conventions (`linspace` takes a count; `arange` takes a step). The question as worded matches the one that **specifies a number of samples**.

### Q5 — axis for columns in 2D

- **Where to look:** [ndarray methods](../1.4-data-foundation-linear-algebra/ndarray-methods.md) — axis conventions.
- **How to think:** Axis 0 indexes the **rows** (going down a column). Axis 1 indexes the **columns** (going across a row). When you `sum(axis=0)` you collapse rows and get one value **per column**.

### Q6 — purpose of `np.array()`

- **Where to look:** [Intro to NumPy](../1.4-data-foundation-linear-algebra/intro-numpy.md).
- **How to think:** `np.array(...)` both creates an array and converts a Python iterable into one. The question hints at "both b and d" — read the options carefully.

### Q7 — number of dimensions

- **Where to look:** [ndarray basics](../1.4-data-foundation-linear-algebra/ndarray-basic.md) — `ndim`.
- **How to think:** "Number of dimensions" has a short, four-letter attribute name. `shape` gives you the size **per dimension**, not the count itself.

### Q8 — generate random numbers

- **Where to look:** [ndarray basics](../1.4-data-foundation-linear-algebra/ndarray-basic.md) — random module.
- **How to think:** Random functions live in the submodule `np.random`. You need to spell out **both** the submodule and the function. `np.random()` alone is not callable.

### Q9 — reshape an array

- **Where to look:** [ndarray basics](../1.4-data-foundation-linear-algebra/ndarray-basic.md) — `reshape`.
- **How to think:** Reshape is available both as a method on the array and as a top-level function. Look for an option that names a working syntax.

### Q10 — what `np.ones()` creates

- **Where to look:** [Intro to NumPy](../1.4-data-foundation-linear-algebra/intro-numpy.md).
- **How to think:** Same pattern as `np.zeros()` — the function name tells you the fill value.

### Q11 — mean of an array

- **Where to look:** [ndarray methods](../1.4-data-foundation-linear-algebra/ndarray-methods.md) — reductions.
- **How to think:** Reductions like `mean`, `sum`, `max` come in **both** flavours: as a method on the array (`arr.mean()`) and as a top-level function (`np.mean(arr)`).

### Q12 — what is broadcasting

- **Where to look:** [ndarray methods](../1.4-data-foundation-linear-algebra/ndarray-methods.md) — broadcasting rules.
- **How to think:** Broadcasting is the **set of rules** NumPy uses to operate on arrays of different shapes without explicitly looping. It is not any single arithmetic operation.

### Q13 — stack arrays vertically

- **Where to look:** [ndarray methods](../1.4-data-foundation-linear-algebra/ndarray-methods.md) — stacking helpers.
- **How to think:** The `h`, `v`, `d` prefixes stand for horizontal, vertical, depth. Pick the prefix that matches "vertical".

### Q14 — what `np.arange()` does

- **Where to look:** [ndarray basics](../1.4-data-foundation-linear-algebra/ndarray-basic.md).
- **How to think:** Mirror of Python's built-in `range`, but returns a NumPy array of evenly spaced values.

### Q15 — maximum value

- **Where to look:** [ndarray methods](../1.4-data-foundation-linear-algebra/ndarray-methods.md).
- **How to think:** Like `mean`, `max` has both method and function forms.

### Q16 — purpose of `np.eye()`

- **Where to look:** [Linear algebra](../1.4-data-foundation-linear-algebra/linear-algebra.md) — identity matrix.
- **How to think:** "Eye" is a pun on the letter **I**, which mathematicians use for the **identity matrix**.

### Q17 — dot product

- **Where to look:** [Linear algebra](../1.4-data-foundation-linear-algebra/linear-algebra.md).
- **How to think:** Same dual-form pattern: top-level function and array method both exist.

### Q18 — what `np.unique()` does

- **Where to look:** [ndarray methods](../1.4-data-foundation-linear-algebra/ndarray-methods.md).
- **How to think:** `unique` finds distinct values **and** returns them **sorted** **and** therefore effectively removes duplicates. The combined option captures all three effects.

### Q19 — transpose a matrix

- **Where to look:** [Linear algebra](../1.4-data-foundation-linear-algebra/linear-algebra.md) — transpose.
- **How to think:** Three notations actually work in NumPy — the attribute `.T`, the top-level `np.transpose()`, and the method `.transpose()`. Pick the option that does not exclude any of them.

### Q20 — purpose of `np.where()`

- **Where to look:** [Boolean indexing](../1.4-data-foundation-linear-algebra/boolean-indexing.md).
- **How to think:** `np.where(cond)` returns **indices** where the condition is true; `np.where(cond, a, b)` does conditional **selection** between `a` and `b`. Both behaviours exist — pick the inclusive option.

---

## Part 5 — Pandas Data Analysis

Lesson root: [1.5 Data analysis with pandas](../1.5-data-analysis-pandas/README.md).

### Q1 — handle missing values

- **Where to look:** [DataFrame](../1.5-data-analysis-pandas/dataframe.md) — missing data handling.
- **How to think:** "NA" is pandas' shorthand for missing. Two of the listed names (`remove`, `delete`, `clean`) are not real pandas methods. The right one is built from `drop` + `na`.

### Q2 — select a single column

- **Where to look:** [DataFrame](../1.5-data-analysis-pandas/dataframe.md) — column access.
- **How to think:** Column access uses bracket notation, like dict lookup: `df[column_name]`. Dot access (`df.col`) also works but only when the column name is a valid identifier.

### Q3 — merge two DataFrames

- **Where to look:** [DataFrame](../1.5-data-analysis-pandas/dataframe.md) — merge / join / concat.
- **How to think:** `concat` is for stacking, `join` joins on the index, `combine` is rarely used. The SQL-style join-on-key operation has the most SQL-like method name.

### Q4 — group data

- **Where to look:** [Function mapping](../1.5-data-analysis-pandas/function-mapping.md) — groupby.
- **How to think:** Pandas borrowed the name straight from SQL's `GROUP BY`. Drop the space and turn it into one word.

### Q5 — sort by values

- **Where to look:** [Sorting & ranking](../1.5-data-analysis-pandas/sorting-ranking.md).
- **How to think:** Pandas has two sort methods — one by values, one by index. The values one has the word `values` in its name.

### Q6 — read a CSV

- **Where to look:** [DataFrame](../1.5-data-analysis-pandas/dataframe.md) — I/O.
- **How to think:** Pandas reader functions all start with `read_…`. Three of the listed names do not exist.

### Q7 — purpose of `df.head()`

- **Where to look:** [DataFrame](../1.5-data-analysis-pandas/dataframe.md) — inspection.
- **How to think:** `head()` returns the **first N rows**, defaulting to 5. It is not just the first row, and it is not column headers.

### Q8 — select multiple columns

- **Where to look:** [DataFrame](../1.5-data-analysis-pandas/dataframe.md) — column selection.
- **How to think:** Passing a **list of column names** inside the brackets returns multiple columns. That means **double** brackets: outer for the selector, inner for the list.

### Q9 — what `df.describe()` does

- **Where to look:** [DataFrame](../1.5-data-analysis-pandas/dataframe.md) — summary methods.
- **How to think:** "Describe" suggests **summarise statistically** — count, mean, std, min, quartiles, max for numeric columns.

### Q10 — rename columns

- **Where to look:** [Reindexing & dropping](../1.5-data-analysis-pandas/reindexing-dropping.md).
- **How to think:** The method is literally called after the operation: re-naming. Other names (`change`, `modify`) are not pandas methods.

### Q11 — purpose of `df.fillna()`

- **Where to look:** [DataFrame](../1.5-data-analysis-pandas/dataframe.md) — missing data.
- **How to think:** `fillna` = "fill NA". It does not remove or count missing values, it **substitutes** them with something.

### Q12 — add a new column

- **Where to look:** [DataFrame](../1.5-data-analysis-pandas/dataframe.md) — column assignment.
- **How to think:** The Pythonic way is **assignment** with bracket notation: `df['new_col'] = values`. No special method is required for the common case.

### Q13 — what `df.info()` shows

- **Where to look:** [DataFrame](../1.5-data-analysis-pandas/dataframe.md).
- **How to think:** `info()` is about **structure and completeness** — index, column names, non-null counts, and dtypes — not statistical values.

### Q14 — select rows by condition

- **Where to look:** [DataFrame](../1.5-data-analysis-pandas/dataframe.md) — boolean indexing.
- **How to think:** Pass a **boolean Series** into the brackets and pandas keeps the rows where it is True. No method call needed for the simple case.

### Q15 — purpose of `value_counts`

- **Where to look:** [Series](../1.5-data-analysis-pandas/series.md) — value counts.
- **How to think:** It counts the occurrences of each **distinct (unique)** value in a Series. Useful for "what categories are in this column and how often?"

### Q16 — save DataFrame to CSV

- **Where to look:** [DataFrame](../1.5-data-analysis-pandas/dataframe.md) — I/O.
- **How to think:** Writers in pandas mirror readers: `read_csv` reads, `to_csv` writes. The `to_*` family handles output.

### Q17 — what `df.shape` returns

- **Where to look:** [DataFrame](../1.5-data-analysis-pandas/dataframe.md) — attributes.
- **How to think:** Same name as NumPy's `shape` — a tuple `(rows, cols)`. It is about **dimensions**, not data types or memory.

### Q18 — sort by index

- **Where to look:** [Sorting & ranking](../1.5-data-analysis-pandas/sorting-ranking.md).
- **How to think:** Sort-by-values uses `sort_values`; sort-by-index uses the matching name with `index` in it.

### Q19 — purpose of `df.apply()`

- **Where to look:** [Function mapping](../1.5-data-analysis-pandas/function-mapping.md).
- **How to think:** `apply` applies a **function** along an axis of the DataFrame (per row or per column). It is not for sorting, formatting, or filtering.

### Q20 — basic info about a DataFrame

- **Where to look:** [DataFrame](../1.5-data-analysis-pandas/dataframe.md).
- **How to think:** Same answer as Q13 — the method whose name is short for "information".

---

## Note on the answer key

The instructor-facing [answer key](./module-assignment-key.md) currently lists only 5 questions per section (and a few of them no longer match the student assignment word-for-word). Filling in keys for the remaining 95 questions is tracked as a separate task; the hints above are written against the student assignment as it stands.
