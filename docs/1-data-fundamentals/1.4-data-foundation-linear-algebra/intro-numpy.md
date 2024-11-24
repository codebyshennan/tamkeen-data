# Introduction to Numpy

## Introduction

Numpy, short for Numerical Python, is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

Many computational and data science packages use Numpy as the main building block. It is a fundamental library for scientific computing in Python.

Some features of Numpy:

- `ndarray`, an efficient multidimensional array providing fast array-oriented arithmetic operations and flexible broadcasting capabilities.
- Mathematical functions for fast operations on entire arrays of data without having to write loops.
- Tools for reading/writing array data to disk and working with memory-mapped files.
- Linear algebra, random number generation, and Fourier transform capabilities.
- A C API for connecting Numpy with libraries written in C, C++, or FORTRAN.

The advantages of using Numpy:

- Numpy internally stores data in a contiguous block of memory, independent of other built-in Python objects. Numpy's library of algorithms written in the C language can operate on this memory without any type checking or other overhead. NumPy arrays also use much less memory than built-in Python sequences (e.g., lists).
- Numpy operations perform complex computations on entire arrays without the need for Python for loops, which can be slow for large sequences. This is called _vectorization_.

![numpy_vs_list](./assets/numpy_vs_python_list.png)

You can install Numpy by using `conda` or `pip`:

```bash
conda install numpy
```

```bash
pip install numpy
```

Then, you can import Numpy as follows:

```python
import numpy as np
```

where `np` is a standard alias for numpy.

To give you an idea of the performance difference, consider a Numpy array of one million integers, and the equivalent Python list:

```python
my_arr = np.arange(1_000_000)
my_list = list(range(1_000_000))
```

Let's multiply each sequence by 2. You can use the `%timeit` magic command to measure the execution time of the code:

```python
%timeit my_arr2 = my_arr * 2
```

```python
%timeit my_list2 = [x * 2 for x in my_list]
```

Numpy operations and algorithms are generally 10 to 100 times faster than their pure Python counterparts and use significantly less memory.
