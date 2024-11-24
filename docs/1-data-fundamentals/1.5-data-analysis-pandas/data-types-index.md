# Data Types and Index

## Data Types (dtypes)

For the most part, pandas uses Numpy arrays and dtypes for Series or individual columns of a DataFrame. NumPy provides support for `float`, `int`, `bool`, `timedelta64[ns]` and `datetime64[ns]` (note that Numpy does not support timezone-aware datetimes).

Pandas also extends Numpy type system, refer to the [documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#basics-dtypes) for more details.

Pandas has two ways to store strings.

- `object` dtype, which can hold any Python object, including strings.
- `StringDtype`, which is dedicated to strings.

Finally, arbitrary objects may be stored using the object dtype, but should be avoided to the extent possible (for performance and interoperability with other libraries and methods).

```python
dft = pd.DataFrame(
    {
        "A": np.random.rand(3),
        "B": 1,
        "C": "foo",
        "D": pd.Timestamp("20010102"),
        "E": pd.Series([1.0] * 3).astype("float32"),
        "F": False,
        "G": pd.Series([1] * 3, dtype="int8")})

dft
```

```python
dft.dtypes
```

On a `Series` object, use the `dtype` attribute:

```python
dft["A"].dtype
```

If a pandas object contains data with multiple dtypes in a single column, the dtype of the column will be chosen to accommodate all of the data types (`object` is the most general).

```python
# these ints are coerced to floats
pd.Series([1, 2, 3, 4, 5, 6.0])
```

```python
# string data forces an `object` dtype
pd.Series([1, 2, 3, 6.0, "foo"])
```

You can use the `astype()` method to explicitly convert dtypes from one to another.

```python
dft["G"].astype("float64")
```

`select_dtypes` method allows you to select columns based on their dtype:

```python
dft.select_dtypes(include=['number'])
```

```python
dft.select_dtypes(include=['number', 'bool'])
```

```python
dft.select_dtypes(include=['number', 'bool'], exclude=['int8'])
```

## Index Objects

Pandas' `Index` objects are responsible for holding the axis labels (a Series' index, a DataFrame's index or a DataFrame's column names) and other metadata (like the axis name or names).

Any array or other sequence of labels used when constructing a Series or DataFrame is internally converted to an Index:

```python
obj = pd.Series(np.arange(3), index=["a", "b", "c"])

index = obj.index

index
```

```python
# slice the index

index[1:]
```

Index objects are immutable and hence can't be modified by the user:

```python
index[1] = "d"
```

You can share Index objects between data structures:

```python
labels = pd.Index(np.arange(3))

labels
```

```python
obj2 = pd.Series([1.5, -2.5, 0], index=labels)

obj2
```

```python
# check if they are the same

obj2.index is labels
```

An Index also behaves like a set:

```python
frame3.columns
```

```python
"Ohio" in frame3.columns
```

```python
2003 in frame3.columns
```

Unlike a set, an Index can contain duplicate labels. Selections with duplicate labels will select all occurrences of that label.

You can refer to the [Index](https://pandas.pydata.org/pandas-docs/stable/reference/indexing.html) documentation for more information.
