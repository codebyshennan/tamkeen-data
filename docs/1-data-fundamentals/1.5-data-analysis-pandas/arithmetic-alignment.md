# Arithmetic and Data Alignment

Similar to Series, for arithmetic computations, data alignment introduces missing values in the label locations that don't overlap.

```python
df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), columns=list("bcd"), index=["Ohio", "Texas", "Colorado"])

df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list("bde"), index=["Utah", "Ohio", "Texas", "Oregon"])
```

```python
df1
```

```python
df2
```

```python
df1 + df2
```

In arithmetic operations between differently indexed objects, you might want to fill with a special value, like 0, when an axis label is found in one object but not the other.

Here is an example where we set a particular value to NA (null) by assigning `np.nan` to it:

```python
df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)), columns=list("abcd"))
df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)), columns=list("abcde"))

df2.loc[1, "b"] = np.nan

```

```python
df1
```

```python
df2
```

Using the add method on df1, pass df2 and an argument to `fill_value`, which substitutes the passed value for any missing values in the operation:

```python
df1.add(df2, fill_value=0)
```

Refer to the [Flexible Binary Operations](https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#matching-broadcasting-behavior) documentation for more information.

## Combining overlapping data sets

A problem occasionally arising is the combination of two similar data sets where _values in one are preferred over the other_.

An example would be two data series representing a particular economic indicator where one is considered to be of _“higher quality”_. However, the lower quality series might extend further back in history or have more complete data coverage.

As such, we would like to combine two DataFrame objects where missing values in one DataFrame are conditionally filled with like-labeled values from the other DataFrame.

```python
df1 = pd.DataFrame(
    {"A": [1.0, np.nan, 3.0, 5.0, np.nan], "B": [np.nan, 2.0, 3.0, np.nan, 6.0]}
)

df2 = pd.DataFrame(
    {
        "A": [5.0, 2.0, 4.0, np.nan, 3.0, 7.0],
        "B": [np.nan, np.nan, 3.0, 4.0, 6.0, 8.0],
    }
)
```

```python
df1
```

```python
df2
```

If we prefer df1 over df2, we can combine the two DataFrames using `combine_first`:

```python
df1.combine_first(df2)
```
