# Basic Syntax and Data Types

## Overview

Syntax and data types are fundamental concepts in any programming language. Syntax is the set of rules that define how code is written, while data types are the types of data that can be used in a program.

## Python Syntax

The most basic Python syntax is the `print()` function. This function is used to display output to the console. For example, to print the string "Hello, World!" to the console, you would use the following code:

```python
print("Hello, World!")
```

If you have a python file (file that ends with `.py`), you can run it from the command line using the `python` command followed by the name of the file. For example, if you have a file named `hello.py` with the following code:

```python
print("Hello, World!")
```

You can run it from the command line (bash in macOS/Linux or command prompt in Windows) using the following command:

```bash
python hello.py
```

### Indentation

Indentation refers to the spaces at the beginning of a code line.

Python uses indentation to define code blocks. This means that the whitespace at the beginning of a line is significant.

For example, in the following code snippet, the `print("Hello")` statement is indented with four spaces, which indicates that it is part of the `if` block:

```python

if 5 > 2:
    print("Five is greater than two!")
```

If you do not use the correct indentation, you will get an `IndentationError`.

The number of spaces is up to you as a programmer, the most common use is four, but it has to be at least one. You just have to make sure to use the same number of spaces in the same block of code, otherwise, you will get an `IndentationError`.

The following code snippet will raise an `IndentationError` because the indentation is inconsistent:

```python
if 5 > 2:
    print("Five is greater than two!")
  print("Hello, World!")
```

### Comments

Comments are used to explain the code and make it more readable. They are ignored by the Python interpreter when the code is executed.

Comments start with a #, and Python will render the rest of the line (after the #) as a comment:

```python
# This is a comment
# print("Welcome")
print("Hello, World!")  # Anything after this is also a comment
```

### Variables

Variables are used to store data values. In Python, variables are created when you assign a value to them.

```python
x = 5
y = "Hello, World!"
```

In the example above, `x` is an _integer (int)_ variable with the value `5`, and `y` is a _string (str)_ variable with the value `"Hello, World!"`.

Variables do not need to be declared with any particular type, and can even change type after they have been set.

```python
x = 5 # x is of type int
x = "Hello, World!" # x is now of type str
print(x)
```

### Variable Names

When naming variables, there are a few rules to follow:

- Variable names must start with a letter or an underscore (`_`).
- Variable names cannot start with a number.
- Variable names can only contain alpha-numeric characters and underscores (`A-z`, `0-9`, and `_`).
- Variable names are case-sensitive (`myVar`, `myvar`, and `MYVAR` are all different variables).
- Variable names should be descriptive and meaningful.
- Variable names should not be the same as Python [keywords](https://www.w3schools.com/python/python_ref_keywords.asp) (e.g., `print`, `if`, `else`, `for`, etc.).

Here are some examples of valid and invalid variable names:

```python
# Valid variable names
myVar = "Hello, World!"
my_var = "Hello, World!"
_my_var = "Hello, World!"
myVar123 = "Hello, World!"

# Invalid variable names
123myVar = "Hello, World!" # Cannot start with a number
my-Var = "Hello, World!" # Cannot contain special characters
my Var = "Hello, World!" # Cannot contain spaces
```

#### Multi Word Variable Names

When naming variables with multiple words, you have a few options:

- Camel Case: `myVariableName`
- Pascal Case: `MyVariableName`
- Snake Case: `my_variable_name`
- Kebab Case: `my-variable-name`

Choose a naming convention that is consistent with the rest of your codebase. In python, the most common convention is to use snake case.

### Assigning Multiple Variables

You can assign values to multiple variables in a single line:

```python
x, y, z = "Orange", "Banana", "Cherry"
print(x)
print(y)
print(z)
```

In the example above, `x` is assigned the value `"Orange"`, `y` is assigned the value `"Banana"`, and `z` is assigned the value `"Cherry"`.

You can also assign the same value to multiple variables in a single line:

```python
x = y = z = "Orange"
print(x, y, z) # You can also print multiple variables in a single line
```

In the example above, `x`, `y`, and `z` are all assigned the value `"Orange"`.

## Primitive Data Types

In Python, data types are divided into two categories: primitive and non-primitive. Primitive data types are the basic data types that are built into the language. The most common primitive data types in Python are:

- **Integers (int):** Whole numbers, e.g., `5`, `-3`, `1000`.
- **Floating-point numbers (float):** Numbers with a decimal point, e.g., `3.14`, `-0.5`, `2.0`.
- **Strings (str):** Sequences of characters enclosed in single or double quotes, e.g., `"Hello, World!"`, `'Python'`.
- **Booleans (bool):** Logical values representing `True` or `False`.
- **NoneType (None):** A special data type representing the absence of a value.

You can use the `type()` function to determine the data type of a variable. For example:

```python
x = 5
y = 3.14
z = "Hello, World!"
a = True
b = None

print(type(x))  # <class 'int'>
print(type(y))  # <class 'float'>
print(type(z))  # <class 'str'>
print(type(a))  # <class 'bool'>
print(type(b))  # <class 'NoneType'>
```

### Type Conversion

Sometimes you may need to convert data from one type to another. Python provides built-in functions for type conversion. The most common type conversion functions are:

- `int()`: Converts a value to an integer.
- `float()`: Converts a value to a floating-point number.
- `str()`: Converts a value to a string.
- `bool()`: Converts a value to a boolean.

For example:

```python
x = 5
y = "10"

# Convert y to an integer
y = int(y)

print(x + y)  # 15
```

### Type Coercion

Type coercion is the automatic conversion of data types by the Python interpreter. Python will automatically convert data types when performing operations that involve different types. For example:

```python
x = 5
y = 3.14

print(x + y)  # 8.14
```

In this example, Python automatically converts the integer `5` to a floating-point number `5.0` before performing the addition operation.

Type coercion can be useful, but it can also lead to unexpected results if you are not careful. It is important to be aware of how Python handles type coercion to avoid potential issues in your code.

### Arithmetic Operators

Arithmetic operators are used to perform mathematical operations in Python. The most common arithmetic operators are:

- **Addition (`+`):** Adds two values.
- **Subtraction (`-`):** Subtracts the second value from the first.
- **Multiplication (`*`):** Multiplies two values.
- **Division (`/`):** Divides the first value by the second.
- **Modulus (`%`):** Returns the remainder of the division.
- **Exponentiation (`**`):\*\* Raises the first value to the power of the second.
- **Floor Division (`//`):** Returns the integer part of the division.

Here are some examples of using arithmetic operators:

```python
x = 10
y = 3

print(x + y)  # 13
print(x - y)  # 7
print(x * y)  # 30
print(x / y)  # 3.3333333333333335
print(x % y)  # 1
print(x ** y)  # 1000
print(x // y)  # 3
```

### Assignment Operators

Assignment operators are used to update the value of a variable. The most common assignment operators are:

- **Assignment (`=`):** Assigns a value to a variable.
- **Addition Assignment (`+=`):** Adds a value to the variable and assigns the result.
- **Subtraction Assignment (`-=`):** Subtracts a value from the variable and assigns the result.
- **Multiplication Assignment (`*=`):** Multiplies the variable by a value and assigns the result.
- **Division Assignment (`/=`):** Divides the variable by a value and assigns the result.

Here are some examples of using assignment operators:

```python
x = 10

x += 5  # Equivalent to x = x + 5
print(x)  # 15

x -= 3  # Equivalent to x = x - 3
print(x)  # 12

x *= 2  # Equivalent to x = x * 2
print(x)  # 24

x /= 4  # Equivalent to x = x / 4
print(x)  # 6.0
```

### Boolean Expressions

Boolean expressions are expressions that evaluate to either `True` or `False`. In Python, boolean expressions are created using comparison operators and logical operators.

#### Comparison Operators

Comparison operators are used to compare two values. The most common comparison operators are:

- **Equal to (`==`):** Returns `True` if the values are equal.
- **Not equal to (`!=`):** Returns `True` if the values are not equal.
- **Greater than (`>`):** Returns `True` if the first value is greater than the second.
- **Less than (`<`):** Returns `True` if the first value is less than the second.
- **Greater than or equal to (`>=`):** Returns `True` if the first value is greater than or equal to the second.
- **Less than or equal to (`<=`):** Returns `True` if the first value is less than or equal to the second.
- **Identity (`is`):** Returns `True` if the variables are the same object.

Here are some examples of using comparison operators:

```python
x = 5
y = 10

print(x == y)  # False
print(x != y)  # True
print(x > y)  # False
print(x < y)  # True
print(x >= y)  # False
print(x <= y)  # True
```

#### Logical Operators

Logical operators are used to combine boolean expressions. The most common logical operators are:

- **AND (`and`):** Returns `True` if both expressions are `True`.
- **OR (`or`):** Returns `True` if at least one expression is `True`.
- **NOT (`not`):** Returns `True` if the expression is `False`.

![Logical Operators](assets/truth-table.png)

Here are some examples of using logical operators:

```python
x = 5
y = 10

print(x > 3 and y > 5)  # True
print(x > 3 and y < 5)  # False
print(x > 3 or y < 5)  # True
print(not x > 0)  # False
```

#### Truthiness and Falsiness

In Python, values have a truthiness or falsiness associated with them. The following values are considered `False`:

- `False`
- `None`
- `0` (integer)
- `0.0` (float)
- `""` (empty string)
- `[]` (empty list)
- `{}` (empty dictionary)
- `()` (empty tuple)
- `set()` (empty set)

All other values are considered `True`.

#### Operator Precedence

Operator precedence determines the order in which operators are evaluated in an expression. Operators with higher precedence are evaluated first. Here is the precedence order from highest to lowest:

1. Parentheses `()`
2. Exponentiation `**`
3. Multiplication `*`, Division `/`, Modulus `%`, Floor Division `//`
4. Addition `+`, Subtraction `-`
5. Comparison Operators `==`, `!=`, `>`, `<`, `>=`, `<=`, `is`
6. Logical Operators `not`, `and`, `or`
7. Assignment Operators `=`, `+=`, `-=`, `*=`, `/=`

You can use parentheses to change the order of evaluation in an expression. For example:

```python
x = 5
y = 10
z = 15

result = x + y * z
print(result)  # 155

result = (x + y) * z
print(result)  # 225
```
