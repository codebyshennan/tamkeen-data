# Functions

## Overview

Functions are reusable blocks of code that perform a specific task. They are used to break down large programs into smaller, more manageable pieces. They can also be reused in different parts of the program.

## Built-in Functions

Built-in functions are functions that are already defined in Python. There is a wide range of built-in functions that can be used to perform various operations. So far, we have used some built-in functions like `print()`, `len()`, `type()`, `sorted()`, `range()`, `enumerate()`, `zip()`, etc. Here are some of the commonly used built-in functions in Python:

- `abs()`: Returns the absolute value of a number.
- `max()`: Returns the largest item in an iterable.
- `min()`: Returns the smallest item in an iterable.
- `sum()`: Returns the sum of all items in an iterable.
- `round()`: Rounds a number to a specified number of decimal places.
- `all()`: Returns `True` if all items in an iterable are `True`.
- `any()`: Returns `True` if any item in an iterable is `True`.

For example, let's use some of these functions:

```python
# abs()
print(abs(-10))  # 10

# max()
print(max([1, 2, 3, 4, 5]))  # 5

# min()
print(min([1, 2, 3, 4, 5]))  # 1

# sum()
print(sum([1, 2, 3, 4, 5]))  # 15

# round()
print(round(3.14159, 2))  # 3.14

# all()
print(all([True, True, True]))  # True
print(all([True, False, True]))  # False

# any()
print(any([True, False, False]))  # True
print(any([False, False, False]))  # False
```

## User-Defined Functions

Now, we will learn how to define our own functions. This is also known as user-defined functions.

### Functions

In Python, you can define a function using the `def` keyword followed by the function name and a pair of parentheses. For example:

```python
def greet():
    print("Hello, World!")
```

In the example above, we defined a function called `greet` that prints "Hello, World!" when called. To call a function, you simply write the function name followed by a pair of parentheses. For example:

```python
greet()
```

This will output:

```
Hello, World!
```

Functions can also take arguments. Arguments are values that are passed to the function when it is called. For example:

```python
def greet(name):
    print(f"Hello, {name}!")
```

In the example above, we defined a function called `greet` that takes an argument `name` and prints "Hello, {name}!" when called. To call this function, you need to pass a value for the `name` argument. For example:

```python
greet("Alice")
```

This will output:

```
Hello, Alice!
```

In the examples above, we defined functions that only print messages, but don't return any values. If we assign the result of calling these functions to a variable, the variable will be `None`. For example:

```python
result = greet("Alice")

print(result)  # None
```

To return a value from a function, you can use the `return` statement. For example:

```python
def add(a, b):
    return a + b
```

In the example above, we defined a function called `add` that takes two arguments `a` and `b` and returns their sum when called. To use the result of this function, you can assign it to a variable. For example:

```python
result = add(3, 5)

print(result)  # 8
```

### Multiple Return Values

You can also return multiple values from a function by separating them with commas. For example:

```python
def add_and_subtract(a, b):
    return a + b, a - b
```

In the example above, we defined a function called `add_and_subtract` that takes two arguments `a` and `b` and returns their sum and difference when called. To use the results of this function, you can assign them to multiple variables. For example:

```python
sum_result, diff_result = add_and_subtract(10, 5)

print(sum_result)  # 15
print(diff_result)  # 5
```

This is the tuple unpacking feature of Python.

### Multiple Return Statements

A function can have multiple `return` statements. The first `return` statement that is executed will exit the function. For example:

```python
def is_even(number):
    if number % 2 == 0:
        return True
    else:
        return False
```

In the example above, we defined a function called `is_even` that takes a number as an argument and returns `True` if the number is even, and `False` otherwise. The function has two `return` statements. The first `return` statement will be executed if the number is even, and the second `return` statement will be executed if the number is odd.

### Default Arguments

We can have default values for function arguments. For example:

```python
def greet(name="World"):
    print(f"Hello, {name}!")
```

In the example above, we defined a function called `greet` that takes an argument `name` with a default value of `"World"`. If you call this function without passing any arguments, it will use the default value. For example:

```python
greet()  # Hello, World!
greet("Alice")  # Hello, Alice!
```

## Variable Scope

Variable scope refers to the visibility of variables within a program. Variables can have different scopes depending on where they are defined. Understanding variable scope is important because it determines where a variable can be accessed and modified.

### Local Scope

Variables that are defined inside a function have local scope. This means that they can only be accessed within the function in which they are defined. For example:

```python
def greet():
    message = "Hello, World!"
    print(message)

greet()
print(message)  # NameError: name 'message' is not defined
```

In the example above, the variable `message` is defined inside the `greet` function and has local scope. It cannot be accessed outside the function.

### Global Scope

Variables that are defined outside of any function have global scope. This means that they can be accessed and modified anywhere in the program. For example:

```python
message = "Hello, World!"

def greet():
    print(message)

greet()
print(message)
```

In the example above, the variable `message` is defined outside the `greet` function and has global scope. It can be accessed and printed inside the function as well as outside the function.

### Modifying Global Variables

You cannot modify a global variable inside a function without explicitly declaring it as a global variable. For example:

```python
count = 0

def increment():
    count += 1

increment()  # UnboundLocalError: local variable 'count' referenced before assignment
```

In the example above, the `increment` function tries to modify the global variable `count` by incrementing it, but it raises an error. To modify a global variable inside a function, you need to declare it as a global variable using the `global` keyword. For example:

```python
count = 0

def increment():
    global count
    count += 1

increment()
print(count)  # 1
```

In the example above, the `global` keyword is used to indicate that the `count` variable is global. This allows the `increment` function to modify the global variable.

## Arbitrary Arguments

In Python, you can define functions that take an arbitrary number of arguments. These are called arbitrary arguments. To define a function with arbitrary arguments, you use an asterisk (`*`) before the parameter name. For example:

```python
def greet(*names):
    for name in names:
        print(f"Hello, {name}!")

greet("Alice", "Bob", "Charlie")
```

In the example above, the `greet` function takes an arbitrary number of arguments and prints a greeting for each name passed to the function.

You can also pass a list or tuple of values to a function with arbitrary arguments by using the unpacking operator (`*`). For example:

```python
names = ["Alice", "Bob", "Charlie"]

greet(*names)
```

This will output:

```
Hello, Alice!
Hello, Bob!
Hello, Charlie!
```

## Keyword Arguments

In Python, you can also define functions that take keyword arguments. Keyword arguments are arguments that are passed to a function with a keyword and a value. To define a function with keyword arguments, you use two asterisks (`**`) before the parameter name. For example:

```python
def greet(**names):
    for name, greeting in names.items():
        print(f"{greeting}, {name}!")

greet(Alice="Hello", Bob="Hi", Charlie="Hey")
```

In the example above, the `greet` function takes keyword arguments and prints a custom greeting for each name passed to the function.

You can also pass a dictionary of values to a function with keyword arguments by using the unpacking operator (`**`). For example:

```python
names = {"Alice": "Hello", "Bob": "Hi", "Charlie": "Hey"}

greet(**names)
```

This will output:

```
Hello, Alice!
Hi, Bob!
Hey, Charlie!
```

## Lambda Functions

In Python, you can define small anonymous functions using the `lambda` keyword, followed by the arguments and a colon (`:`) that separates the arguments from the expression. The expression is evaluated and returned when the lambda function is called. There is no `return` statement in a lambda function because the expression is implicitly returned.

Lambda functions can take any number of arguments, but they can only have one expression.

Example with one argument:

```python
double = lambda x: x * 2

print(double(5))  # 10
```

Example with multiple arguments:

```python
add = lambda a, b: a + b

print(add(3, 5))  # 8
```

Lambda functions are often used with built-in functions like `map()` and `filter()` to apply a function to a sequence of elements or filter elements based on a condition.

For example, you can use a lambda function with `map()` to double each element in a list:

```python
numbers = [1, 2, 3, 4, 5]

doubled_numbers = list(map(lambda x: x * 2, numbers))

print(doubled_numbers)  # [2, 4, 6, 8, 10]
```

You can also use a lambda function with `filter()` to filter out even numbers from a list:

```python
numbers = [1, 2, 3, 4, 5]

even_numbers = list(filter(lambda x: x % 2 == 0, numbers))

print(even_numbers)  # [2, 4]
```
