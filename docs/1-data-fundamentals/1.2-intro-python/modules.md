# Modules

## Overview

In Python, a module is a file that contains Python code. Modules are used to organize and reuse code in Python programs. You can import modules to access their functions, classes, and variables in your code.

Python comes with a standard library that includes a wide range of modules for different purposes, such as working with files, dates, math, and regular expressions. You can also create your own custom modules to encapsulate related code and functionality.

## Importing Modules

To use a module in Python, you need to import it into your code. You can import a module using the `import` statement followed by the module name.

```python
import math

print(math.pi)  # 3.141592653589793
```

We import the `math` module, which provides mathematical functions and constants. We then access the value of `math.pi` to print the mathematical constant π.

You can also import specific functions or variables from a module using the `from` statement.

```python
from math import sqrt

print(sqrt(25))  # 5.0
```

We import the `sqrt` function from the `math` module and use it to calculate the square root of 25.

## Creating Custom Modules

You can create your own custom modules in Python by defining functions, classes, and variables in a Python file. To use a custom module, you need to save the Python file with a `.py` extension and import it into your code.

For example, you can create a custom module named `utils.py` with the following content:

```python
def greet(name):
    return f"Hello, {name}!"
```

You can then import the `greet` function from the `utils` module and use it in your code as follows:

```python
from utils import greet

print(greet("Alice"))  # Hello, Alice!
```

We import the `greet` function from the `utils` module and use it to print a greeting message for the name "Alice".
