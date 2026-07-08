# Data Foundation with NumPy

**After this submodule:** you can use the lessons linked below and complete the exercises that match **Data Foundation with NumPy** in your course schedule.

> **Time needed:** 4-5 hours to complete this module

## Why Learn NumPy?

Ever wondered how data scientists handle millions of numbers efficiently? Or how they perform complex calculations at lightning speed? Welcome to NumPy - your superpower for handling data in Python!

> **Note:** NumPy (short for "Numerical Python") is the foundation that powers almost all data science tools in Python. Once you master NumPy, learning Pandas and other libraries becomes much easier!

> **Contributors:** Authoring standards: `docs/meta/DOCUMENTATION_GUIDELINES.md` (`meta/` is excluded from the Jekyll build).

## Prerequisites

Before starting this module, you should have:

- Completed the "Introduction to Python" module
- Basic understanding of Python lists and data types
- Familiarity with basic math operations
- Completed the "Introduction to Statistics" module (helpful but not required)
- Comfortable with arrays or matrices from math class (we'll review if needed)

> **Tip:** Think of NumPy arrays like super-powered lists. If you understand Python lists, you're already halfway there!

### Video

<div class="video-embed">
<iframe width="560" height="315" src="https://www.youtube.com/embed/QUT1VHiLmmI" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

*freeCodeCamp, Python NumPy tutorial for beginners*

---

### The Power of NumPy

Think of NumPy as a turbocharger for Python:

- 100x faster than regular Python lists
- Efficient memory usage
- Powerful mathematical operations
- Perfect for data science and AI

---

### Real-World Applications

NumPy is everywhere:

- Data Analysis
- Machine Learning
- Image Processing
- Financial Analysis

## Understanding data types (nominal through ratio)

Before you average or model, ask what **kind of measurement** you have. The four **levels of measurement** below are standard vocabulary; they tell you which summaries are meaningful.

| Level | Order? | Equal steps? | True zero? | Examples | Sensible summaries |
| ----- | ------ | -------------- | ---------- | -------- | ------------------- |
| **Nominal** | No |, | No | Country codes, blood type | Counts, proportions, mode |
| **Ordinal** | Yes | Not guaranteed equal | No | Likert scales, education band | Median, ranks; means need care |
| **Interval** | Yes | Yes | No | Celsius, calendar years | Means, differences; ratios often misleading |
| **Ratio** | Yes | Yes | Yes | Height, weight, revenue | Full arithmetic, ratios ("twice as much") |

**Nominal** values are labels: blue is not "greater than" red. You can count categories and find the mode, but an "average color" is nonsense.

**Ordinal** values can be sorted (mild, medium, hot), but the **distance** between ranks may be uneven, five stars are not necessarily "equal steps" in satisfaction. Medians and percentiles are often safer than means.

**Interval** scales have equal-sized steps, but **zero is arbitrary** (0°C is not "no heat"). Differences matter; ratios like "twice as hot" usually do not.

**Ratio** scales have a **true zero** (no money, no height). All arithmetic operations are on the table, including ratios.

### Categorical vs continuous

**Categorical** variables take distinct buckets (often integers or strings): product ID, survey choice. **Continuous** variables can vary smoothly across a range: time, distance, temperature measured finely. Both appear in NumPy and pandas; plots and statistics should match the type (bar charts vs histograms, counts vs density).

## Quick reference (compact)

```
Level      Order   Steps    Zero    Example
Nominal    no - no      Colors
Ordinal    yes     uneven   no      Star ratings
Interval   yes     equal    no      °C
Ratio      yes     equal    yes     Height (cm)
```

## What We'll Learn

Get ready to master NumPy through these exciting topics:

1. **Introduction to NumPy**
   - Fast calculations
   - Efficient arrays

2. **NumPy ndarray**
   - Multi-dimensional arrays
   - Data organization

3. **ndarray Basics**
   - Creating arrays
   - Basic operations

4. **Boolean Indexing**
   - Filtering data
   - Conditional selection

5. **ndarray Methods**
   - Useful functions
   - Data manipulation

6. **Linear Algebra**
   - Matrix operations
   - Mathematical tools

> **Pro Tip:** Understanding data types is important because it determines what operations you can perform on your data!

## What You'll Be Able to Do After This Module

By the end of this module, you'll be able to:

- Create and manipulate NumPy arrays efficiently
- Perform fast mathematical operations on large datasets
- Use boolean indexing to filter data
- Apply NumPy methods for data analysis
- Understand basic linear algebra operations
- Work with multi-dimensional arrays
- Write efficient code that processes data quickly

> **Tip:** NumPy operations are much faster than regular Python loops. Once you get comfortable with NumPy, you'll never want to go back to slow Python loops for data processing!

## Next Steps

After completing this module, you'll move on to:

1. **Data Analysis with Pandas** - Build on NumPy to work with structured data
2. **Data Wrangling** - Use NumPy and Pandas together to clean and transform data
3. **Data Visualization** - Visualize NumPy arrays and analysis results

> **Note:** NumPy is used behind the scenes in Pandas, so understanding NumPy will make Pandas much easier to learn. Take your time to master these fundamentals!
