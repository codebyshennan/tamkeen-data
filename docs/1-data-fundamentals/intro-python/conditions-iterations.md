# Conditions and Iterations

## Overview

Conditions and iterations are fundamental concepts in any programming language. Conditions are used to check if a certain condition is met, and iterations are used to repeat a certain block of code multiple times.

## If, Elif, Else

Conditional statements are used to execute different blocks of code based on certain conditions. In Python, the `if`, `elif`, and `else` statements are used to create conditional statements.

### If Statement

The `if` statement is used to execute a block of code if a condition is `True`. The syntax of the `if` statement is as follows:

```python
if condition:
    # code block
```

The code block is executed only if the condition is `True`. If the condition is `False`, the code block is skipped.

For example:

```python
x = 5

if x > 0:
    print("x is positive")
```

In this example, the code block `print("x is positive")` is executed because the condition `x > 0` is `True`.

### Elif Statement

The `elif` statement is used to check multiple conditions after the `if` statement. The syntax of the `elif` statement is as follows:

```python
if condition1:
    # code block
elif condition2:
    # code block
```

The `elif` statement is only executed if the previous conditions are `False` and the current condition is `True`.

For example:

```python
x = 0

if x > 0:
    print("x is positive")
elif x < 0:
    print("x is negative")
elif x == 0:
    print("x is zero")
```

In this example, the code block `print("x is zero")` is executed because the previous conditions `x > 0` and `x < 0` are `False`, and the current condition `x == 0` is `True`.

### Else Statement

The `else` statement is used to execute a block of code if all the previous conditions are `False`. The syntax of the `else` statement is as follows:

```python
if condition:
    # code block
else:
    # code block
```

The `else` statement is only executed if all the previous conditions are `False`.

For example:

```python
x = -5

if x > 0:
    print("x is positive")
else:
    print("x is not positive")
```

In this example, the code block `print("x is not positive")` is executed because the condition `x > 0` is `False`.

In the elif example before, we can also replace the last elif statement with an else statement:

```python
x = 0

if x > 0:
    print("x is positive")
elif x < 0:
    print("x is negative")
else:
    print("x is zero")
```

In this example, the code block `print("x is zero")` is executed because the previous conditions `x > 0` and `x < 0` are `False`, and the `else` statement is executed.

### Shorthand If

You can also write a shorthand `if` statement, which is a one-liner version of the `if` statement. The syntax of the shorthand `if` statement is as follows:

```python
if condition: # code block
```

For example:

```python
x = 5

if x > 0: print("x is positive")
```

### Shorthand If Else

You can also write a shorthand `if else` statement, which is a one-liner version of the `if else` statement. The syntax of the shorthand `if else` statement is as follows:

```python
value_if_true if condition else value_if_false
```

For example:

```python
x = 5

print("x is positive") if x > 0 else print("x is not positive")
```

In this example, the code block `print("x is positive")` is executed if the condition `x > 0` is `True`, otherwise the code block `print("x is not positive")` is executed.

## While Loop

Loops are used to execute a block of code multiple times. In Python, the `while` loop is used to execute a block of code as long as a condition is `True`.

The syntax of the `while` loop is as follows:

```python
while condition:
    # code block
```

The code block is executed repeatedly as long as the condition is `True`. If the condition is `False`, the loop is terminated, and the program continues with the next statement after the loop.

For example:

```python
x = 0

while x < 5:
    print(x)
    x += 1
```

In this example, the loop will print the numbers `0`, `1`, `2`, `3`, and `4` because the condition `x < 5` is `True` for these values of `x`.

### Break Statement

The `break` statement is used to exit a loop prematurely. When the `break` statement is encountered, the loop is terminated, and the program continues with the next statement after the loop.

For example:

```python
x = 0

while x < 5:
    print(x)
    if x == 2:
        break
    x += 1
```

In this example, the loop will print the numbers `0`, `1`, and `2` because the `break` statement is encountered when `x == 2`.

### Continue Statement

The `continue` statement is used to skip the rest of the code block and continue with the next iteration of the loop. When the `continue` statement is encountered, the program jumps back to the beginning of the loop and evaluates the condition again.

For example:

```python
x = 0

while x < 5:
    x += 1
    if x == 2:
        continue
    print(x)
```

In this example, the loop will print the numbers `1`, `3`, `4`, and `5` because the `continue` statement is encountered when `x == 2`.

### Else Statement

The `else` statement can be used in a loop to execute a block of code when the loop condition is `False`. The syntax of the `else` statement in a loop is as follows:

```python
while condition:
    # code block
else:
    # code block
```

The `else` statement is only executed when the loop condition is `False`. If the loop is terminated prematurely using the `break` statement, the `else` statement is not executed.

For example:

```python
x = 0

while x < 5:
    print(x)
    x += 1
else:
    print("Loop finished")
```

In this example, the loop will print the numbers `0`, `1`, `2`, `3`, and `4`, and then print "Loop finished" because the loop condition `x < 5` is `False` when `x == 5`.

## For Loop

In Python, the `for` loop is used to iterate over a sequence of elements, such as a range of numbers, a string, a list, or a dictionary.

The syntax of the `for` loop is as follows:

```python
for element in sequence:
    # code block
```

The `for` loop iterates over each element in the sequence and executes the code block for each element. The loop continues until all elements in the sequence have been processed.

For example:

```python
fruits = ["apple", "banana", "cherry"]

for fruit in fruits:
    print(fruit)
```

In this example, the loop will print each element in the `fruits` list.

### Range Function

The `range()` function is commonly used in `for` loops to generate a sequence of numbers. The `range()` function takes up to three arguments: `start`, `stop`, and `step`. The `start` argument specifies the starting value of the sequence, the `stop` argument specifies the end value of the sequence (not inclusive), and the `step` argument specifies the increment between each value in the sequence.

For example:

```python
for i in range(5):
    print(i)
```

In this example, the loop will print the numbers `0`, `1`, `2`, `3`, and `4`.

You can also specify the `start`, `stop`, and `step` arguments in the `range()` function:

```python
for i in range(1, 10, 2):
    print(i)
```

In this example, the loop will print the numbers `1`, `3`, `5`, `7`, and `9`.

### Break Statement

The `break` statement is used to exit a loop prematurely. When the `break` statement is encountered, the loop is terminated, and the program continues with the next statement after the loop.

For example:

```python
for i in range(5):
    print(i)
    if i == 2:
        break
```

In this example, the loop will print the numbers `0`, `1`, and `2` because the `break` statement is encountered when `i == 2`.

### Continue Statement

The `continue` statement is used to skip the rest of the code block and continue with the next iteration of the loop. When the `continue` statement is encountered, the program jumps back to the beginning of the loop and evaluates the condition again.

For example:

```python
for i in range(5):
    if i == 2:
        continue
    print(i)
```

In this example, the loop will print the numbers `0`, `1`, `3`, and `4` because the `continue` statement is encountered when `i == 2`.

### Else Statement

The `else` statement can be used in a loop to execute a block of code when the loop completes normally (i.e., without encountering a `break` statement). The syntax of the `else` statement in a loop is as follows:

```python
for element in sequence:
    # code block
else:
    # code block
```

The `else` statement is only executed if the loop completes normally without encountering a `break` statement.

For example:

```python
for i in range(5):
    print(i)
else:
    print("Loop completed normally")
```

In this example, the loop will print the numbers `0`, `1`, `2`, `3`, and `4`, and then print `Loop completed normally`.

### Nested For Loops

You can nest `for` loops inside each other to iterate over multiple sequences. The inner loop is executed for each iteration of the outer loop.

For example:

```python
fruits = ["apple", "banana", "cherry"]
colors = ["red", "yellow", "purple"]

for fruit in fruits:
    for color in colors:
        print(fruit, color)
```

In this example, the inner loop will iterate over each element in the `colors` list for each element in the `fruits` list.

### Looping Through Strings and Dictionaries

You can also use the `for` loop to iterate over strings and dictionaries.

#### Strings

You can iterate over each character in a string using a `for` loop:

```python
for char in "hello":
    print(char)
```

In this example, the loop will print each character in the string `"hello"`.

#### Dictionaries

You can iterate over each key in a dictionary using a `for` loop:

```python
person = {"name": "Alice", "age": 30, "city": "New York"}

for key in person:
    print(key, person[key])
```

In this example, the loop will print each key-value pair in the `person` dictionary.

You can also iterate over each key-value pair in a dictionary using the `items()` method. This also makes use of tuple unpacking:

```python
for key, value in person.items():
    print(key, value)
```

In this example, the loop will print each key-value pair in the `person` dictionary.

### Enumerate Function

The `enumerate()` function can be used in a `for` loop to get both the index and value of each element in a sequence. The `enumerate()` function returns a tuple containing the index and value of each element.

For example:

```python
fruits = ["apple", "banana", "cherry"]

for index, fruit in enumerate(fruits):
    print(index, fruit)
```

In this example, the loop will print the index and value of each element in the `fruits` list.

### Zip Function

The `zip()` function can be used in a `for` loop to iterate over multiple sequences simultaneously. The `zip()` function combines the elements of multiple sequences into tuples and returns an iterator of tuples.

For example:

```python
fruits = ["apple", "banana", "cherry"]
colors = ["red", "yellow", "purple"]

for fruit, color in zip(fruits, colors):
    print(fruit, color)
```

In this example, the loop will print each element in the `fruits` list paired with the corresponding element in the `colors` list.

## Comprehensions

Comprehensions are a concise way to create sequences, such as lists, dictionaries, sets, and generators, in Python. They allow you to create a new sequence by transforming or filtering an existing sequence. Comprehensions are more readable and expressive than traditional loops and can help you write more efficient and maintainable code.

### List Comprehensions

List comprehensions are used to create lists in Python. The syntax of a list comprehension is as follows:

```python
new_list = [expression for item in iterable if condition]
```

The list comprehension iterates over each item in the iterable and applies the expression to create a new list. The condition is optional and can be used to filter the items in the iterable.

For example:

```python
numbers = [1, 2, 3, 4, 5]

squares = [x ** 2 for x in numbers]

print(squares)  # [1, 4, 9, 16, 25]
```

In this example, the list comprehension `x ** 2` is applied to each element in the `numbers` list to create a new list of squares.

### Dictionary Comprehensions

Dictionary comprehensions are used to create dictionaries in Python. The syntax of a dictionary comprehension is as follows:

```python
new_dict = {key: value for item in iterable if condition}
```

The dictionary comprehension iterates over each item in the iterable and applies the expression to create a new dictionary. The condition is optional and can be used to filter the items in the iterable.

For example:

```python
fruits = ["apple", "banana", "cherry"]

fruit_lengths = {fruit: len(fruit) for fruit in fruits}

print(fruit_lengths)  # {'apple': 5, 'banana': 6, 'cherry': 6}
```

In this example, the dictionary comprehension `len(fruit)` is applied to each element in the `fruits` list to create a new dictionary of fruit lengths.

You can also use dictionary comprehensions to create dictionaries from other dictionaries:

```python
students = {"Alice": 25, "Bob": 30, "Charlie": 35}

adult_students = {name: age for name, age in students.items() if age >= 30}

print(adult_students)  # {'Bob': 30, 'Charlie': 35}
```

### Generator Comprehensions

Generator is a type of iterable that generates values on-the-fly. Instead of storing all the values in memory, a generator produces values one at a time, which can be more memory-efficient for large sequences.

Generator comprehensions are used to create generators in Python. The syntax of a generator comprehension is similar to a list comprehension, but it uses parentheses instead of square brackets:

```python
new_generator = (expression for item in iterable if condition)
```

The generator comprehension iterates over each item in the iterable and applies the expression to create a new generator. The condition is optional and can be used to filter the items in the iterable.

For example:

```python
numbers = [1, 2, 3, 4, 5]

squares_generator = (x ** 2 for x in numbers)

print(squares_generator)  # <generator object <genexpr> at 0x7f8b1c7b3d60>
print(list(squares_generator))  # [1, 4, 9, 16, 25]
```

In this example, the generator comprehension `x ** 2` is applied to each element in the `numbers` list to create a new generator of squares.
