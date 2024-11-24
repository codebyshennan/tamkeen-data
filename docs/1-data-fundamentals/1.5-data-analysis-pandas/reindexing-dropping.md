# Reindexing and Dropping Index

## Reindexing

`reindex` is the fundamental data alignment method in pandas. It is used to create a new object with the data conformed to a new index.

```python
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])

obj
```

`reindex` can be used to rearrange the data according to the new index, introducing missing values if any index values were not already present:

```python
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])

obj2
```

You can reindex the (row) index or columns of a dataframe:

```python
frame = pd.DataFrame(np.arange(9).reshape((3, 3)),
                        index=['a', 'c', 'd'],
                        columns=['Ohio', 'Texas', 'California'])

frame
```

```python
frame2 = frame.reindex(['a', 'b', 'c', 'd'])

frame2
```

```python
# reindex the columns

states = ["Texas", "Utah", "California"]

frame.reindex(columns=states)
```

```python
# another way to reindex the columns

frame.reindex(states, axis="columns")
```

The `method` option allows us to do interpolation or filling of values when reindexing. For example, we can fill the missing values with the last known value (forward fill):

```python
obj3 = pd.Series(["blue", "purple", "yellow"], index=[0, 2, 4])

obj3
```

```python
obj3.reindex(np.arange(6), method="ffill")
```

This is useful for filling in missing values when reindexing ordered data like time series.

## Dropping Index

You can drop entries from an axis using the `drop` method. It will return a new object with the indicated value or values deleted from an axis:

```python
obj = pd.Series(np.arange(5.), index=["a", "b", "c", "d", "e"])

obj
```

```python
new_obj = obj.drop("c")

new_obj
```

```python
obj.drop(["d", "c"])
```

With DataFrame, index values can be deleted from either axis:

```python
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                        index=["Ohio", "Colorado", "Utah", "New York"],
                        columns=["one", "two", "three", "four"])
```

```python
data
```

The default `drop` method is to drop rows (`axis="index"` or `axis=0`).

```python
data.drop(["Colorado", "Ohio"])
```

You can drop columns by specifying `axis="columns"` or `axis=1`:

```python
data.drop("two", axis=1)
```
