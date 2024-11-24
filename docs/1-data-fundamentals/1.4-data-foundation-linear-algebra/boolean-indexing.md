# Boolean Indexing

Let's consider an example where we have an array of names with duplicates, and an array of scores (for 2 subjects) that correspond to each name.

```python
names = np.array(["Bob", "Joe", "Will", "Bob", "Will", "Joe", "Joe"])
scores = np.array(
    [[75, 80], [85, 90], [95, 100], [100, 77], [85, 92], [95, 80], [72, 80]]
)
```

If we want to select all the rows with the corresponding name 'Bob'. Like arithmetic operations, comparisons (such as `==`) with arrays are also vectorized. Thus, comparing `names` with the string 'Bob' yields a boolean array.

```python
names == "Bob"
```

This boolean array can be passed when indexing the array.

```python
scores[names == "Bob"]
```

You can mix boolean indexing with other slicing and indexing methods.

```python
scores[names == "Bob", 1]
```

To select everything but 'Bob', you can either use `!=` or negate the condition using `~`.

```python
names != "Bob"
```

```python
~(names == "Bob")
```

```python
scores[names != "Bob"]
```

The `~` operator can be useful when you want to invert a boolean array referenced by a variable.

```python
cond = names == "Bob"
cond
```

```python
scores[~cond]
```

You can select two or more names by combining multiple boolean conditions. Use boolean arithmetic operators like `&` (and) and `|` (or).

```python
mask = (names == "Bob") | (names == "Will")
mask
```

```python
scores[mask]
```

```python
scores > 80
```

You can also set the values based on these boolean arrays. For example, to set all scores less than 80 to 70:

```python
scores[scores < 80] = 70
```

```python
scores
```

To select a subset of the rows in a particular order, you can simply pass a list or ndarray of integers specifying the desired order.

```python
arr = np.zeros((8, 4))

for i in range(8):
    arr[i] = i

arr
```

```python
arr[[4, 3, 0, 6]]
```

Negative indices select rows from the end.

```python
arr[[-3, -5, -7]]
```
