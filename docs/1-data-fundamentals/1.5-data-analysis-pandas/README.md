# Data Analysis with Pandas

## Introduction to Pandas

Pandas, short for "Python Data Analysis Library" or "Panel Data", is a library for data manipulation and analysis. It is built on top of `Numpy` and is designed to work with `Numpy` arrays. Pandas is a library that provides high-performance, easy-to-use data structures and data analysis tools. It is designed for quick and easy data manipulation, aggregation, and visualization.

Pandas is often used in conjunction with numerical computing tools like `Numpy` and `SciPy`, analytical libraries like `statsmodels` and `scikit-learn`, and data visualization libraries like `Matplotlib` and `Seaborn`. While pandas adopts many coding idioms from `Numpy`, the biggest difference is that pandas is designed for working with tabular or heterogeneous data. Numpy, by contrast, is best suited for working with homogeneous numerical array data.

Some of the key features of Pandas are:

- Fast and efficient DataFrame object with default and customized indexing.
- Tools for loading data into in-memory data objects from different file formats.
- Data alignment and integrated handling of missing data.
- Reshaping and pivoting of data sets.
- Label-based slicing, indexing and subsetting of large data sets.
- Group by data for aggregation and transformations.
- High performance merging and joining of data.
- Time series functionality.

## Data Structures

Pandas has two main data structures:

- `Series`: 1-dimensional labeled array capable of holding any data type (integers, strings, floating point numbers, Python objects, etc.). The axis labels are collectively referred to as the index.
- `DataFrame`: 2-dimensional labeled data structure with columns of potentially different types. You can think of it like a spreadsheet or SQL table, or a dictionary of Series objects. It is generally the most commonly used pandas object.

## Chapter Outline

This chapter will cover the following topics:

1. Series
2. DataFrame
3. Data Types and Index
4. Reindexing and Dropping Index
5. Arithmetic and Data Alignment
6. Function Application and Mapping
7. Sorting and Ranking
