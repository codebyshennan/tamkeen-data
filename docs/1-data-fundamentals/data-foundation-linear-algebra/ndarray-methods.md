# ndarray Methods

## Reshaping and Transposing Arrays

Arrays have the `reshape` method to change the shape of a given array to a new shape that has the same number of elements. For example, you can reshape a 1D array to a 2D array with 2 rows and 3 columns.

```python
arr = np.arange(15).reshape((3, 5))
arr
```

Arrays have the `transpose` method for rearranging data. For a 2D array, `transpose` will return a new view on the data with axes swapped.

```python
arr.transpose()
```

The `T` attribute is a shortcut for `transpose`.

```python
arr.T
```

## Universal Functions

A universal function, or `ufunc`, is a function that performs element-wise operations on data in ndarrays. You can think of them as fast vectorized wrappers for simple functions that take one or more scalar values and produce one or more scalar results.

```python
arr = np.arange(10)
arr
```

Calculate the square root of each element in the array:

```python
np.sqrt(arr)
```

Calculate the exponential of each element in the array:

```python
np.exp(arr)
```

These are referred to as unary ufuncs. Others, such as `add` or `maximum`, take 2 arrays (thus, binary ufuncs) and return a single array as the result.

```python
x = np.array([3, 7, 15, 5, 12])
y = np.array([11, 2, 4, 6, 8])

np.maximum(x, y)
```

You can refer to the [Numpy documentation](https://numpy.org/doc/stable/reference/ufuncs.html#available-ufuncs) for a list of all available universal functions.

## Conditional Logic

If you want to evaluate all elements in an array based on a condition, you can use `np.where`, a vectorized version of the ternary expression `x if condition else y`.

Suppose we had a boolean array and two arrays of values:

```python
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
```

If we wanted to take a value from `xarr` whenever the corresponding value in `cond` is `True`; otherwise, take the value from `yarr`:

```python
np.where(cond, xarr, yarr)
```

The second and third arguments to `numpy.where` don’t need to be arrays; one or both of them can be scalars. A typical use of `where` in data analysis is to produce a new array of values based on another array.

## Array Methods

You can generate a random array using the `np.random` module. The `randn` function returns a sample (or samples) from the "standard normal" distribution. A standard normal distribution is a normal distribution with a mean of 0 and standard deviation of 1.

Here we generate a random 3x4 array of samples from the standard normal distribution.

```python
arr = np.random.randn(3, 4)
arr
```

Calculate the average:

```python
arr.mean()
```

You can also use the universal function `np.mean`:

```python
np.mean(arr)
```

Calculate the sum:

```python
arr.sum()
```

You can also provide an optional argument `axis` that specifies the axis along which the statistic is computed, resulting in an array with one fewer dimension.

```python
arr.mean(axis=1)
```

```python
arr.mean(axis=0)
```

`axis=1` means "compute across the columns," whereas `axis=0` means "compute down the rows."

Refer to the diagram again for the illustration on axes.

![ndarray](./assets/numpy_ndarray.png)

> Compute the sum across the columns of `arr`.

For boolean arrays, `any` tests whether one or more values in an array is `True`, while `all` checks if every value is `True`.

```python
bools = np.array([False, False, True, False])
```

```python
bools.any()
```

```python
bools.all()
```

Like Python’s built-in list type, NumPy arrays can be sorted with the `sort` method. Note that this method sorts a data array _in-place_, meaning that the array contents are rearranged rather than a new array being created.

```python
arr = np.random.randn(8)
arr
```

```python
arr.sort()
```
