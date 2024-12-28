# Data Foundation with Numpy

## Introduction to Data Types

Understanding data types is crucial in data science, as it determines the types of analyses that can be applied. Data types are often classified into **nominal, ordinal, interval, and ratio** levels of measurement, each with distinct characteristics. Additionally, data can also be broadly categorized into **categorical** and **continuous** types, depending on how it is measured and analyzed.

## Nominal Data

**Nominal data** is the simplest level of measurement, involving categories or names that do not have a specific order. These are labels or identifiers used to distinguish between items.

- **Characteristics**:

  - Categories are **mutually exclusive** (each item fits in one category only).
  - No inherent order among categories.
  - Examples: Gender (male, female), Types of cuisine (Italian, Chinese, Mexican), Colors (red, blue, green).

- **Summary**: Nominal data is **qualitative** and typically **categorical**.

### Mathematical Operations on Nominal Data

- **Permissible**: Counting, mode (most frequent category).
- **Not Permissible**: Mean, median, or standard deviation, as there is no numeric meaning.

## Ordinal Data

**Ordinal data** has ordered categories, allowing us to rank items, but the intervals between the ranks are not equal or specified.

- **Characteristics**:

  - Categories have a meaningful order, indicating a rank or level.
  - Differences between ranks are not standardized.
  - Examples: Customer satisfaction ratings (satisfied, neutral, dissatisfied), Military ranks (lieutenant, captain, major), Education levels (high school, college, graduate).

- **Summary**: Ordinal data is also **qualitative** and **categorical**, but with an inherent order.

### Mathematical Operations on Ordinal Data

- **Permissible**: Counting, mode, median (for ranking).
- **Not Permissible**: Mean, as the intervals between ranks are not equal.

## Interval Data

**Interval data** includes ordered categories with equal intervals between values, but it lacks a true zero point, meaning zero does not indicate the absence of the variable.

- **Characteristics**:

  - Equal intervals between values allow for meaningful comparisons of differences.
  - Lacks a true zero, so ratios are not meaningful.
  - Examples: Temperature in Celsius or Fahrenheit, Calendar years (2020, 2021), IQ scores.

- **Summary**: Interval data is **quantitative** and often treated as **continuous**.

### Mathematical Operations on Interval Data

- **Permissible**: Addition and subtraction, mean, median, standard deviation.
- **Not Permissible**: Ratios (e.g., twice as much) since zero does not imply "none."

## Ratio Data

**Ratio data** has all the properties of interval data, with the addition of a meaningful zero point, which indicates the absence of the measured attribute.

- **Characteristics**:

  - Ordered with equal intervals between values and a true zero, making ratios meaningful.
  - Allows for comparisons of magnitude (e.g., twice as heavy, half as tall).
  - Examples: Height, Weight, Duration of time, Income.

- **Summary**: Ratio data is **quantitative** and can be treated as **continuous**.

### Mathematical Operations on Ratio Data

- **Permissible**: All arithmetic operations, including addition, subtraction, multiplication, and division, as well as mean, median, and standard deviation.

## Summary of Data Types

| Level    | Type         | Order | Equal Intervals | True Zero | Examples                    |
| -------- | ------------ | ----- | --------------- | --------- | --------------------------- |
| Nominal  | Qualitative  | No    | No              | No        | Colors, types of cuisine    |
| Ordinal  | Qualitative  | Yes   | No              | No        | Satisfaction ratings, ranks |
| Interval | Quantitative | Yes   | Yes             | No        | Temperature, IQ scores      |
| Ratio    | Quantitative | Yes   | Yes             | Yes       | Height, weight, income      |

## Categorical vs. Continuous Data

### Categorical Data

**Categorical data** describes attributes that fall into distinct groups or categories, often based on the nominal or ordinal levels of measurement.

- **Nominal and Ordinal data** are often classified as categorical.
- **Properties**:
  - Divides data into groups.
  - Limited to descriptive or frequency-based analysis.
  - Examples: Gender, education level, satisfaction ratings.

### Continuous Data

**Continuous data** represents measurements on a continuous scale and includes interval and ratio data.

- **Interval and Ratio data** are considered continuous.
- **Properties**:
  - Can take any value within a range.
  - Supports a full range of mathematical operations.
  - Examples: Weight, temperature, time.

## Chapter Outline

This chapter will cover the following topics:

1. Introduction to Numpy
2. Numpy ndarray
3. ndarray Basic
4. Boolean Indexinig
5. ndarray Methods
6. Linear Algebra
