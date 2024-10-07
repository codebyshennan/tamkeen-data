# Data Structures

## Overview

Data structures are used to store and organize data in a computer program. They provide a way to manage and manipulate data efficiently. Python provides several built-in data structures, including lists, tuples, dictionaries, and sets. We will also discuss strings, as a special type of primitive data types.

## String

Strings are sequences of characters enclosed in single or double quotes. They are one of the most common data types in Python and are used to represent text data.

```python
x = "Hello, World!"
y = 'Python'
```

### Quotes inside Strings

If you need to include quotes inside a string, you can use the other type of quotes to enclose the string. For example:

```python
x = "He said, 'Hello, World!'"
y = 'She said, "Python"'
```

### Multiline Strings

If you need to create a string that spans multiple lines, you can use triple quotes (`'''` or `"""`) to enclose the string. For example:

```python
x = '''This is a
multiline
string.'''
```

### Accessing Characters

![indexing](assets/python-index.png)

#### Indexing

You can access individual characters in a string using indexing. In Python, strings are zero-indexed, which means that the first character is at index `0`, the second character is at index `1`, and so on.

```python
x = "Hello, World!"

print(x[0])  # H
print(x[7])  # W
```

You can also use negative indexing to access characters from the end of the string. In negative indexing, the last character is at index `-1`, the second-to-last character is at index `-2`, and so on.

```python
print(x[-1])  # !
print(x[-6])  # W
```

#### Slicing

You can also return a range of characters from a string using slicing. Slicing is done by specifying the start and end indices separated by a colon (`:`). For example:

```python
print(x[7:12])  # World
```

If you omit the start index, Python will start from the beginning of the string. If you omit the end index, Python will go to the end of the string. For example:

```python
print(x[:5])  # Hello
print(x[7:])  # World
```

There's an optional third parameter in slicing called the step. The step specifies the increment between the characters in the slice. For example:

```python
print(x[2:12:2]) # lo Wrd
print(x[::2])  # Hlo ol!
```

You can reverse a string by using a negative step.

```python
print(x[::-1])  # !dlroW ,olleH
```

### String Concatenation

You can concatenate (combine) strings using the `+` operator. For example:

```python
x = "Hello"
y = "World"

z = x + ", " + y + "!"

print(z)  # Hello, World!
```

You can also use the `*` operator to repeat a string a certain number of times. For example:

```python
x = "Hello"

y = x * 3

print(y)  # HelloHelloHello
```

### String Methods

Python provides many built-in methods for working with strings. Here are some common string methods:

- `upper()`: Converts the string to uppercase.
- `lower()`: Converts the string to lowercase.
- `strip()`: Removes whitespace from the beginning and end of the string.
- `replace()`: Replaces a substring with another substring.
- `split()`: Splits the string into a list of substrings based on a delimiter.
- `join()`: Joins a list of strings into a single string.

Here are some examples of using string methods:

```python
x = "Hello, World!"

print(x.upper())  # HELLO, WORLD!
print(x.lower())  # hello, world!
print(x.strip())  # Hello, World!
print(x.replace("Hello", "Hi"))  # Hi, World!
print(x.split(","))  # ['Hello', ' World!']
```

Note that string methods do not modify the original string; they return a new string with the modified content. If you want to modify the original string, you need to assign the result back to the original variable.

```python
x = "Hello, World!"

x = x.upper()

print(x)  # HELLO, WORLD!
```

There is a built-in function called `len()` that returns the length of a string.

> Note the difference between a function and a method. A function is a standalone piece of code that can be called with arguments. A method is a function that belongs to an object and is called using the dot notation.

```python
x = "Hello, World!"

print(len(x))  # 13
```

### String Formatting

String formatting allows you to create dynamic strings by inserting variables into a string. There are several ways to format strings in Python, but the most common method is using the `format()` method.

You can use curly braces `{}` as placeholders in a string and pass variables to the `format()` method to replace the placeholders with the variable values. For example:

```python
name = "Alice"
age = 30

x = "My name is {} and I am {} years old.".format(name, age)

print(x)  # My name is Alice and I am 30 years old.
```

You can also use curly braces with indices to specify the order of the variables passed to the `format()` method. For example:

```python
x = "My name is {1} and I am {0} years old.".format(age, name)

print(x)  # My name is Alice and I am 30 years old.
```

#### f-Strings

f-strings are a more recent and convenient way to format strings in Python. You can prefix a string with `f` or `F` and use curly braces `{}` to insert variables directly into the string. For example:

```python
name = "Alice"
age = 30

x = f"My name is {name} and I am {age} years old."

print(x)  # My name is Alice and I am 30 years old.
```

You can also perform operations inside the curly braces.

```python
num1 = 10
num2 = 20

x = f"The sum of {num1} and {num2} is {num1 + num2}"

print(x)  # The sum of 10 and 20 is 30
```

#### Format Specifiers

You can also use format specifiers to format the output of variables in a string.

For example, you can specify the width of the field and the alignment.

```python
name = "Alice"

x = f"Hello, {name:>10}"

print(x)  # Hello,      Alice
```

You can also specify the number of decimal places for a floating-point number.

```python
pi = 3.14159

x = f"The value of pi is {pi:.2f}"

print(x)  # The value of pi is 3.14
```

For integers, you can specify the number of digits and padding.

```python
num = 42

x = f"The answer is {num:05d}"

print(x)  # The answer is 00042
```

## List

Lists are one of the most versatile data structures in Python. They are used to store collections of items, such as numbers, strings, or other objects. Lists are mutable, which means that you can change the elements they contain after the list is created.

When defining a list in Python, you enclose the elements in square brackets (`[]`) and separate them with commas (`,`). For example:

```python
# Define a list of numbers
numbers = [1, 2, 3, 4, 5]

# Define a list of strings
fruits = ["apple", "banana", "cherry"]
```

You can also create an empty list by using empty square brackets or the `list()` constructor:

```python
# Create an empty list
empty_list = []
empty_list = list()
```

The `list()` constructor can also convert a list:

```python
# Create a list from a string
chars = list("hello")

print(chars)  # ['h', 'e', 'l', 'l', 'o']
```

Operators like `+` and `*` work on lists as well. For example:

```python
numbers1 = [1, 2, 3]
numbers2 = [4, 5, 6]

# Concatenate two lists
numbers = numbers1 + numbers2

print(numbers)  # [1, 2, 3, 4, 5, 6]

# Repeat a list
numbers = numbers1 * 3

print(numbers)  # [1, 2, 3, 1, 2, 3, 1, 2, 3]
```

### Accessing Elements

#### Indexing

You can access individual elements in a list using indexing. In Python, lists are zero-indexed, which means that the first element is at index `0`, the second element is at index `1`, and so on.

```python
numbers = [1, 2, 3, 4, 5]

print(numbers[0])  # 1
print(numbers[2])  # 3
```

You can also use negative indexing to access elements from the end of the list. In negative indexing, the last element is at index `-1`, the second-to-last element is at index `-2`, and so on.

```python
print(numbers[-1])  # 5
print(numbers[-3])  # 3
```

#### Slicing

You can extract a sublist (slice) from a list using slicing. Slicing allows you to specify a range of indices to extract elements from the list.

```python
print(numbers[1:4])  # [2, 3, 4]
```

If you omit the start index, Python will start from the beginning of the list. If you omit the end index, Python will go to the end of the list.

```python
print(numbers[:3])  # [1, 2, 3]
print(numbers[2:])  # [3, 4, 5]
```

You can also specify a step value to extract elements at regular intervals.

```python
print(numbers[1:5:2])  # [2, 4]
print(numbers[::2])  # [1, 3, 5]
```

### Modifying Elements

You can modify elements in a list by assigning new values to specific indices. Lists are mutable, so you can change the elements they contain after the list is created.

```python
numbers = [1, 2, 3, 4, 5]

numbers[2] = 10

print(numbers)  # [1, 2, 10, 4, 5]
```

### List Methods

Python provides a variety of methods for working with lists. Here are some common list methods:

- `append()`: Adds an element to the end of the list.
- `insert()`: Inserts an element at a specific index.
- `remove()`: Removes the first occurrence of a value from the list.
- `pop()`: Removes an element at a specific index and returns it.
- `extend()`: Adds the elements of another list to the end of the list.
- `index()`: Returns the index of the first occurrence of a value.

```python
numbers = [1, 2, 3, 4, 5]

# Append an element to the end of the list
numbers.append(6)

print(numbers)  # [1, 2, 3, 4, 5, 6]

# Insert an element at index 2
numbers.insert(2, 10)

print(numbers)  # [1, 2, 10, 3, 4, 5, 6]

# Remove the first occurrence of 3
numbers.remove(3)

print(numbers)  # [1, 2, 10, 4, 5, 6]

# Remove and return the element at index 3
element = numbers.pop(3)

print(element)  # 4
print(numbers)  # [1, 2, 10, 5, 6]

# Add the elements of another list
numbers.extend([7, 8, 9])

print(numbers)  # [1, 2, 10, 5, 6, 7, 8, 9]

# Return the index of the first occurrence of 10
index = numbers.index(10)

print(index)  # 2
```

Just like in strings, you can use the `len()` function to get the length of a list:

```python
numbers = [1, 2, 3, 4, 5]

print(len(numbers))  # 5
```

### Check membership

You can check if an element is present in a list using the `in` operator. For example:

```python
numbers = [1, 2, 3, 4, 5]

print(3 in numbers)  # True
print(6 in numbers)  # False
```

### Sorting Lists

You can sort a list using the `sort()` method, which sorts the list in place. For example:

```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

numbers.sort()

print(numbers)  # [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]
```

You can also use the `sorted()` function to create a new sorted list without modifying the original list. For example:

```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

sorted_numbers = sorted(numbers)

print(sorted_numbers)  # [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]
```

### List to String

You can convert a list of strings to a single string using the `join()` method. For example:

```python
fruits = ["apple", "banana", "cherry"]

fruits_string = ", ".join(fruits)

print(fruits_string)  # apple, banana, cherry
```

### Copying Lists

When you assign a list to a new variable, you are creating a reference to the original list, not a copy of the list. This means that changes made to the new list will affect the original list. To create a copy of a list, you can use the `copy()` method or the `list()` constructor.

```python
numbers = [1, 2, 3, 4, 5]

# Create a reference to the original list
numbers_copy = numbers

# Modify the copy
numbers_copy[2] = 10

print(numbers)  # [1, 2, 10, 4, 5]

# Create a copy of the original list
numbers_copy = numbers.copy()

# Modify the copy
numbers_copy[3] = 20

print(numbers)  # [1, 2, 10, 4, 5]
print(numbers_copy)  # [1, 2, 10, 20, 5]
```

## Tuple

Tuples are another data structure in Python that are similar to lists. They are used to store collections of items, such as numbers, strings, or other objects. However, tuples are immutable, which means that you cannot change the elements they contain after the tuple is created.

When defining a tuple in Python, you enclose the elements in parentheses `()` and separate them with commas `,`. For example:

```python
# Define a tuple of numbers
numbers = (1, 2, 3, 4, 5)

# Define a tuple of strings
fruits = ("apple", "banana", "cherry")
```

You can also create an empty tuple by using empty parentheses or the `tuple()` constructor. Tuples with a single element must have a trailing comma to distinguish them from parentheses:

```python
# Create an empty tuple
empty_tuple = ()
empty_tuple = tuple()

# Tuple with a single element
single_tuple = (1,)
```

The `tuple()` constructor can also convert a tuple:

```python
# Create a tuple from a string
chars = tuple("hello")

print(chars)  # ('h', 'e', 'l', 'l', 'o')
```

Just like lists, operators like `+` and `*` work on tuples as well.

```python
numbers1 = (1, 2, 3)
numbers2 = (4, 5, 6)

# Concatenate two tuples
numbers = numbers1 + numbers2

print(numbers)  # (1, 2, 3, 4, 5, 6)

# Repeat a tuple
numbers = numbers1 * 3

print(numbers)  # (1, 2, 3, 1, 2, 3, 1, 2, 3)
```

### Accessing Elements

Indexing and slicing work the same way for tuples as they do for lists.

```python
numbers = (1, 2, 3, 4, 5)

print(numbers[0])  # 1
print(numbers[2])  # 3
print(numbers[-1])  # 5
print(numbers[:3])  # (1, 2, 3)
print(numbers[2:])  # (3, 4, 5)
```

### Modifying Elements

Since tuples are immutable, you cannot change the elements they contain. If you try to modify a tuple, you will get an error:

```python
numbers = (1, 2, 3, 4, 5)

numbers[0] = 10  # TypeError: 'tuple' object does not support item assignment
```

### Tuple Methods

Tuples have fewer methods than lists because they are immutable. However, there are a few methods that you can use with tuples:

- `count()`: Returns the number of times a specified value occurs in the tuple.
- `index()`: Returns the index of the first occurrence of a specified value.

```python
numbers = (1, 2, 3, 4, 5, 3)

print(numbers.count(3))  # 2
print(numbers.index(4))  # 3
```

### Unpacking Tuples

You can unpack a tuple by assigning its elements to multiple variables. This is useful when you want to assign the elements of a tuple to individual variables:

```python
numbers = (1, 2, 3)

x, y, z = numbers

print(x)  # 1
print(y)  # 2
print(z)  # 3
```

You can also use the `*` operator to unpack the remaining elements of a tuple into a list:

```python
numbers = (1, 2, 3, 4, 5)

x, *y, z = numbers

print(x)  # 1
print(y)  # [2, 3, 4]
print(z)  # 5
```

## Set

A set is a collection of unique elements in Python. Sets are unordered, which means that the elements do not have a specific order. Sets are mutable, which means that you can add or remove elements from a set after it is created.

When defining a set in Python, you enclose the elements in curly braces `{}` and separate them with commas `,`. For example:

```python
# Define a set of numbers
numbers = {1, 2, 3, 4, 5}

# Define a set of strings
fruits = {"apple", "banana", "cherry"}
```

You can also create an empty set by using empty curly braces or the `set()` constructor. However, you cannot create an empty set using empty curly braces because they are used to define an empty dictionary:

```python
# Create an empty set
empty_set = set()
```

The `set()` constructor can also convert a list or tuple to a set:

```python
# Create a set from a list
numbers = set([1, 2, 3, 4, 5])

print(numbers)  # {1, 2, 3, 4, 5}

# If the list contains duplicate elements, they will be removed in the set
numbers = set([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5])

print(numbers)  # {1, 2, 3, 4, 5}
```

### Accessing Elements

Sets are unordered, which means that you cannot access elements in a set using indexing.

### Modifying Elements

Sets are mutable, which means that you can add or remove elements from a set after it is created.

#### Adding Elements

You can add elements to a set using the `add()` method:

```python
fruits = {"apple", "banana", "cherry"}

fruits.add("orange")

print(fruits)  # {'apple', 'banana', 'cherry', 'orange'}

# If the element already exists in the set, it will not be added again
fruits.add("apple")

print(fruits)  # {'apple', 'banana', 'cherry', 'orange'}
```

You can also add multiple elements to a set using the `update()` method:

```python
fruits = {"apple", "banana", "cherry"}

fruits.update(["apple", "pear", "grape"])

print(fruits)  # {'apple', 'banana', 'cherry', 'grape', 'pear'}
```

#### Removing Elements

You can remove elements from a set using the `remove()` method. If the element does not exist in the set, you will get an error:

```python
fruits = {"apple", "banana", "cherry"}

fruits.remove("banana")

print(fruits)  # {'apple', 'cherry'}
```

### Set Operations

Sets support various operations, such as union, intersection, difference, and symmetric difference.

![set-operations](assets/python-set.webp)

#### Union

The union of two sets contains all the elements from both sets:

```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}

union = set1 | set2

print(union)  # {1, 2, 3, 4, 5}
```

You can also use the `union()` method to perform the union operation:

```python
union = set1.union(set2)

print(union)  # {1, 2, 3, 4, 5}
```

#### Intersection

The intersection of two sets contains the elements that are common to both sets:

```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}

intersection = set1 & set2

print(intersection)  # {3}
```

You can also use the `intersection()` method to perform the intersection operation:

```python
intersection = set1.intersection(set2)

print(intersection)  # {3}
```

#### Difference

The difference between two sets contains the elements that are in the first set but not in the second set:

```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}

difference = set1 - set2

print(difference)  # {1, 2}
```

You can also use the `difference()` method to perform the difference operation:

```python
difference = set1.difference(set2)

print(difference)  # {1, 2}
```

#### Symmetric Difference

The symmetric difference between two sets contains the elements that are in either set but not in both sets:

```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}

symmetric_difference = set1 ^ set2

print(symmetric_difference)  # {1, 2, 4, 5}
```

You can also use the `symmetric_difference()` method to perform the symmetric difference operation:

```python
symmetric_difference = set1.symmetric_difference(set2)

print(symmetric_difference)  # {1, 2, 4, 5}
```

## Dictionary

Dictionaries are another data structure in Python that are used to store collections of items. Unlike lists and tuples, which are indexed by a range of numbers, dictionaries are indexed by keys. Keys can be any _immutable_ type, such as strings, numbers, or tuples. While the values in a dictionary can be of any type, the keys must be unique within a dictionary.

When defining a dictionary in Python, you enclose the key-value pairs in curly braces `{}` and separate them with commas `,`. Each key-value pair is separated by a colon `:`. For example:

```python
# Define a dictionary of student names and their ages
students = {"Alice": 20, "Bob": 21, "Charlie": 22}

# Define an empty dictionary
empty_dict = {}
```

You can also create a dictionary using the `dict()` constructor:

```python
# Create a dictionary using the dict() constructor
students = dict(Alice=20, Bob=21, Charlie=22)

print(students)  # {'Alice': 20, 'Bob': 21, 'Charlie': 22}
```

### Accessing Elements

Dictionaries are indexed by keys, which means that you can access elements in a dictionary using the keys. If the key exists in the dictionary, you can access the corresponding value using the key. For example:

```python
print(students["Alice"])  # 20
print(students["Bob"])    # 21
```

If the key does not exist in the dictionary, you will get a `KeyError`:

```python
print(students["Eve"])  # KeyError: 'Eve'
```

To avoid this error, you can use the `get()` method, which returns `None` if the key does not exist in the dictionary:

```python
print(students.get("Eve"))  # None
```

You can also specify a default value to return if the key does not exist in the dictionary:

```python
print(students.get("Eve", 0))  # 0
```

### Modifying Elements

Dictionaries are mutable, which means that you can add, update, or remove elements from a dictionary after it is created.

#### Adding Elements

You can add elements to a dictionary by assigning a value to a new key:

```python
students["Eve"] = 23

print(students)  # {'Alice': 20, 'Bob': 21, 'Charlie': 22, 'Eve': 23}
```

#### Updating Elements

You can update elements in a dictionary by assigning a new value to an existing key:

```python
students["Alice"] = 21

print(students)  # {'Alice': 21, 'Bob': 21, 'Charlie': 22, 'Eve': 23}
```

#### Removing Elements

You can remove elements from a dictionary using the `del` keyword:

```python
del students["Eve"]

print(students)  # {'Alice': 21, 'Bob': 21, 'Charlie': 22}
```

You can also use the `pop()` method to remove an element from a dictionary and return its value:

```python
age = students.pop("Bob")

print(age)      # 21
print(students)  # {'Alice': 21, 'Charlie': 22}
```

### Dictionary Methods

Dictionaries have several methods that you can use to work with them:

- `keys()`: Returns a view of the keys in the dictionary.
- `values()`: Returns a view of the values in the dictionary.
- `items()`: Returns a view of the key-value pairs in the dictionary.
- `get()`: Returns the value for a specified key. If the key does not exist, it returns `None`.
- `pop()`: Removes the element with the specified key and returns its value.
- `popitem()`: Removes the last inserted key-value pair from the dictionary and returns it.
- `update()`: Updates the dictionary with the specified key-value pairs.

```python
students = {"Alice": 20, "Bob": 21, "Charlie": 22}

# Get the keys in the dictionary
print(students.keys())  # dict_keys(['Alice', 'Bob', 'Charlie'])

# Get the values in the dictionary
print(students.values())  # dict_values([20, 21, 22])

# Get the key-value pairs in the dictionary
print(students.items())  # dict_items([('Alice', 20), ('Bob', 21), ('Charlie', 22])

# Remove the last inserted key-value pair
key, value = students.popitem()

print(key)    # Charlie
print(value)  # 22

# Update the dictionary with new key-value pairs
students.update({"David": 23, "Eve": 24})

print(students)  # {'Alice': 20, 'David': 23, 'Eve': 24}
```

### Copying a Dictionary

When you assign a dictionary to a new variable, you are creating a reference to the original dictionary, not a copy of it. This means that if you modify the new dictionary, the original dictionary will also be modified. To create a copy of a dictionary, you can use the `copy()` method:

```python
students = {"Alice": 20, "Bob": 21, "Charlie": 22}

# Create a reference to the original dictionary
students_copy = students

# Modify the copy
students_copy["Alice"] = 25

print(students)        # {'Alice': 25, 'Bob': 21, 'Charlie': 22}

# Create a copy of the original dictionary
students_copy = students.copy()

# Modify the copy
students_copy["Bob"] = 26

print(students)        # {'Alice': 25, 'Bob': 21, 'Charlie': 22}
print(students_copy)  # {'Alice': 25, 'Bob': 26, 'Charlie': 22}
```
