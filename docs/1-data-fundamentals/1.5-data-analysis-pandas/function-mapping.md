# Function Application and Mapping

NumPy ufuncs (element-wise array methods) work fine with pandas objects:

```python
frame = pd.DataFrame(np.random.randn(4, 3), columns=list("bde"), index=["Utah", "Ohio", "Texas", "Oregon"])

frame
```

```python
np.abs(frame)
```

You can apply a function on one-dimensional arrays to each column or row of a DataFrame using the `apply` method:

```python
def f1(x):
    return x.max() - x.min()

frame.apply(f1)
```

Here the function `f1`, which computes the difference between the maximum and minimum of a Series, is invoked once on each column in frame (default behavior). The result is a Series having the columns of frame as its index.

If you check the documentation of `apply`, the default value for `axis` is `"index"` or `0`, which means the function is applied on each column.

If you pass `axis="columns"` or `axis=1` to `apply`, the function will be invoked once per row instead. A helpful way to think about this is as "apply across the columns":

```python
frame.apply(f1, axis="columns")
```

Many of the most common array statistics (like sum and mean) are DataFrame methods, so using apply is not necessary. We will discuss more about this in the next lesson.

Element-wise Python functions can be used, too. Suppose you wanted to compute a formatted string from each floating-point value in `frame`. You can do this with `applymap`:

```python
def my_format(x):
    return f"{x:.2f}"
```

```python
frame.applymap(my_format)
```
