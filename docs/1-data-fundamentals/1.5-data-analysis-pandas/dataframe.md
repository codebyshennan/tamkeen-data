# DataFrame

A DataFrame represents a tabular, spreadsheet-like data structure containing an ordered collection of columns, each of which can be a different value type (numeric, string, boolean, etc.). The DataFrame has both a row and column index; it can be thought of as a dict of Series (all sharing the same index).

![dataframe](./assets/dataframe.png)

One of the most common ways to create a DataFrame is from a dict of equal-length lists or NumPy arrays:

```python
data = {"state": ["Ohio", "Ohio", "Ohio", "Nevada", "Nevada", "Nevada"],
        "year": [2000, 2001, 2002, 2001, 2002, 2003],
        "pop": [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)

frame
```

Inspect the shape:

```python
frame.shape
```

Select the first 5 rows:

```python
frame.head()
```

Select the last 5 rows:

```python
frame.tail()
```

If you specify a sequence of columns, the DataFrame’s columns will be arranged in that order:

```python
pd.DataFrame(data, columns=["year", "state", "pop"])
```

If you pass a column that isn’t contained in the dictionary, it will appear with missing values:

```python
frame2 = pd.DataFrame(data, columns=["year", "state", "pop", "debt"])

frame2
```

Both the `index` and `columns` are `Index` objects.

```python
frame2.index
```

```python
frame2.columns
```

A column in a DataFrame can be retrieved as a Series either by dict-like notation or by dot attribute notation:

```python
frame2["state"]
```

or

```python
frame2.state
```

Columns can be modified by assignment. For example, the empty 'debt' column could be assigned a scalar value or an array of values:

```python
frame2["debt"] = 16.5

frame2
```

```python
frame2["debt"] = np.arange(6.)

frame2
```

When assigning lists or arrays to a column, the value’s length must match the length of the DataFrame. If you assign a Series, it will be instead conformed exactly to the DataFrame’s index, inserting missing values in any holes:

```python
val = pd.Series([-1.2, -1.5, -1.7], index=["two", "four", "five"])

frame2["debt"] = val

frame2
```

Assigning a column that doesn’t exist will create a new column, for instance, let's create a boolean column that indicates whether the state is eastern:

```python
frame2["eastern"] = frame2.state == "Ohio"

frame2
```

You can remove columns by using the `del` method.

```python
del frame2["eastern"]
```

You can also create a DataFrame from a nested dict of dicts. Pandas will interpret the outer dict keys as the columns and the inner keys as the row indices:

```python
frame3 = pd.DataFrame({"Ohio": {2000: 1.5, 2001: 1.7, 2002: 3.6}, "Nevada": {2001: 2.4, 2002: 2.9}})

frame3
```

Let's transpose the DataFrame (similar to a NumPy array transpose):

```python
frame3.T
```

We can also set the `name` attributes for `index` and `columns`. Unlike Series, DataFrame does not have a `name` attribute.

```python
frame3.index.name = "year"
frame3.columns.name = "state"

frame3
```

You can return the data contained in a DataFrame as a 2D Numpy `ndarray`:

```python
frame3.to_numpy()
```
