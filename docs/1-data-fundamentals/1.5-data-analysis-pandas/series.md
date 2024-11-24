# Series

A series is a one-dimensional array-like object containing a sequence of values (of similar types to NumPy types) and an associated array of data labels, called its index.

The simplest Series is formed from only an array of data:

```python
obj = pd.Series([5, 6, -3, 2])

obj
```

The string representation of a Series displayed interactively shows the `index` on the left and the `values` on the right.

Since we did not specify an index for the data, a default one consisting of the integers 0 through N - 1 (where N is the length of the data) is created.

You can get the array representation and index object of the Series via its `array` and `index` attributes, respectively:

```python
obj.array
```

```python
obj.index
```

Like Numpy array, `shape` attribute returns a tuple with the shape of the data:

```python
obj.shape
```

You can also create a Series with a label index identifying each data point:

```python
obj2 = pd.Series([4, 7, -5, 3], index=["d", "b", "a", "c"])

obj2.index
```

## Indexing and Selection

You can use labels in the index when selecting single values or a set of values:

```python
obj2["b"]
```

Use the same method to update the value for a label:

```python
obj2["d"] = 6

obj2
```

Or index multiple values by passing a list of labels:

```python
obj2[["c", "a", "d"]]
```

## Numpy-like Functions or Operations

You can use Numpy functions or NumPy-like operations, such as filtering with a boolean array, scalar multiplication, or applying math functions.

Filtering (also known as boolean indexing):

```python
obj2[obj2 > 0]
```

Arithmetic operations:

```python
obj2 * 2
```

Applying numpy function:

```python
np.exp(obj2)
```

You can also think about a Series as a fixed-length, ordered `dictionary`, where the keys are the index and the values are the data:

```python
"b" in obj2
```

```python
"e" in obj2
```

## Series from/to Dictionary

You can create a Series from a dictionary, or vice versa:

```python
sdata = {"Ohio": 35000, "Texas": 71000, "Oregon": 16000, "Utah": 5000}
obj3 = pd.Series(sdata)

obj3
```

To convert back to a dictionary:

```python
obj3.to_dict()
```

The index in the resulting Series will respect the order of the keys in the dictionary. You can override this by passing an index with the dictionary keys in the order you want them to appear in the Series:

```python
states = ["California", "Ohio", "Oregon", "Texas"]
obj4 = pd.Series(sdata, index=states)

obj4
```

Since no value for "California" was found, it appears as `NaN` (not a number) which is considered in pandas to mark missing or NA values. Since "Utah" was not included in states, it is excluded from the resulting object.

"Missing", "NA" or "null" can be used interchangeably to refer to missing data. The `isna` and `notna` functions in pandas could be used to detect missing data:

```python
pd.isna(obj4)
```

```python
pd.notna(obj4)
```

or using the instance method for Series:

```python
obj4.isna()
```

## Alignment via Index

A useful Series feature for many applications is that it automatically aligns differently-indexed data in arithmetic operations:

```python
obj3
```

and

```python
obj4
```

If we sum both, the result will be a Series containing the union of the two indexes, with values in the non-overlapping regions set to `NaN`.

```python
obj3 + obj4
```

Both the Series object itself and its index have a `name` attribute, which will be useful later.

Here, the index represents the states and the values represent the population of each state. Let's give a name to each of them:

```python
obj4.name = "population"
obj4.index.name = "state"

obj4
```

You can alter the index in-place by assignment. Remember our previous series:

```python
obj
```

Let's change the index:

```python
obj.index = ["Bob", "Steve", "Jeff", "Ryan"]

obj
```
