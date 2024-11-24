# Sorting and Ranking

## Sorting

Sorting the data by some criterion is another important built-in operation. To sort lexicographically by row or column index, use the `sort_index` method, which returns a new, sorted object:

```python
obj = pd.Series(np.arange(4), index=['d', 'a', 'b', 'c'])

obj
```

```python
obj.sort_index()
```

With a DataFrame, you can sort by index on either axis:

```python
frame = pd.DataFrame(np.arange(8).reshape((2, 4)), index=["three", "one"], columns=["d", "a", "b", "c"])

frame
```

Again, the default behavior is to sort by row `index`. You can sort by column by passing `axis=1` or `axis="columns"`:

```python
frame.sort_index()
```

```python
frame.sort_index(axis=1)
```

Sorting is in ascending order by default, but can be reversed by passing `ascending=False`.

```python
frame.sort_index(axis="columns", ascending=False)
```

To sort by values, use `sort_values`:

```python
obj = pd.Series([4, 7, -3, 2])

obj.sort_values()
```

Any missing values are sorted to the end of the Series by default:

```python
obj = pd.Series([4, np.nan, 7, np.nan, -3, 2])

obj.sort_values()
```

Missing values can be sorted to the start instead by using the `na_position` option:

```python
obj.sort_values(na_position="first")
```

When sorting a DataFrame, you can use the data in one or more columns as the sort keys. To do so, pass one or more column names to `sort_values`:

```python
frame = pd.DataFrame({"b": [4, 7, -3, 2], "a": [0, 1, 0, 1]})

frame
```

```python
frame.sort_values("b")
```

You can also sort by multiple columns:

```python
frame.sort_values(["a", "b"])
```

To sort different columns in different orders, pass a list of booleans specifying whether or not each column should be sorted in ascending order:

```python
frame.sort_values(["a", "b"], ascending=[False, True])
```

## Ranking

Ranking assigns ranks from one through the number of valid data points in an array, starting from the lowest value.

The `rank` methods for Series and DataFrame are the place to look; by default, `rank` breaks ties by assigning each group the mean rank:

```python
obj = pd.Series([7, -5, 7, 4, 2, 4, 0, 4])

obj.rank()
```

Ranks can also be assigned according to the order in which they’re observed in the data:

```python
obj.rank(method="first")
```

Ranking in descending order:

```python
obj.rank(ascending=False)
```

DataFrame can compute ranks over the rows or the columns:

```python
frame = pd.DataFrame({"b": [4.3, 7, -3, 2], "a": [0, 1, 0, 1], "c": [-2, 5, 8, -2.5]})

frame
```

The defaull is to rank down the rows:

```python
frame.rank()
```

To rank by column instead, pass `axis="columns"` or `axis=1`:

```python
frame.rank(axis="columns")
```
