# Classes and Objects

## Overview

Classes are a fundamental concept in object-oriented programming. In object-oriented programming, classes are used to model real-world entities and define their properties and behaviors. Python is an object-oriented programming language, which means that it supports classes and objects. A class is a blueprint for creating objects, and an object is an instance of a class.

## Classes

In Python, you can define a class using the `class` keyword followed by the class name. For example:

```python
class Person:
    pass
```

In the example above, we defined a class called `Person`. The name of the class should be in CamelCase, with the first letter of each word capitalized.

The `pass` statement is used as a placeholder to indicate that the class is empty. We will learn how to define properties and methods inside a class to model the attributes and behaviors of the real-world entity that the class represents.

You can create an object from a class by calling the class name followed by a pair of parentheses. This is called instantiation.

```python
person = Person()
```

In the example above, we created an object called `person` from the `Person` class. The object `person` is an instance of the `Person` class.

### Constructors

A constructor is a special method in a class that is used to initialize the object. In Python, the constructor method is called `__init__`. The constructor method is automatically called when an object is created from a class.

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
```

In the example above, we defined a constructor method `__init__` in the `Person` class. The constructor takes two parameters `name` and `age` and initializes the properties `self.name` and `self.age` with the values passed to the constructor. Properties defined in the constructor are also called instance variables.

The `self` parameter is a reference to the current instance of the class.

You can create an object from the `Person` class and pass values to the constructor as arguments. For example:

```python
person = Person("Alice", 30)

print(person.name)  # Alice
print(person.age)  # 30
```

In the example above, we created an object called `person` from the `Person` class and passed the values `"Alice"` and `30` to the constructor. The object `person` now has two instance variables `name` and `age` with the values `"Alice"` and `30`, respectively.

You can modify the properties of an object by accessing them using the dot notation. For example:

```python
person.age = 31

print(person.age)  # 31
```

## Methods

A method is a function that is defined inside a class. Methods are used to define the behaviors of the objects created from the class. There are three types of methods in Python classes:

- Instance methods
- Class methods
- Static methods

### Instance Methods

Instance methods are methods that are defined inside a class and are called on an instance of the class. Instance methods take `self` as the first parameter, which is a reference to the current instance of the class. Instance methods can access and modify the instance variables of the class. To define an instance method, you use the `def` keyword followed by the method name and the `self` parameter. For example:

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")

    def celebrate_birthday(self):
        self.age += 1
        print(f"Happy birthday, {self.name}! You are now {self.age} years old.")

    def change_name(self, new_name):
        self.name = new_name

person = Person("Alice", 30)
person.greet()  # Hello, my name is Alice and I am 30 years old.
person.celebrate_birthday()  # Happy birthday, Alice! You are now 31 years old.
person.change_name("Bob")
person.greet()  # Hello, my name is Bob and I am 31 years old.
```

### The `__str__` Method

The `__str__` method is a special instance method that is used to return a string representation of an object. The `__str__` method is called when you use the `str()` function or the `print()` function on an object. To define the `__str__` method, you use the `def` keyword followed by the method name `__str__` and the `self` parameter. The `__str__` method should return a string representation of the object. For example:

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"Person(name={self.name}, age={self.age})"

person = Person("Alice", 30)
print(person)  # Person(name=Alice, age=30)
```

### Class Variables and Methods

Class variables are variables that are defined inside a class but outside any method. Class variables are shared among all instances of the class. Class variables are accessed using the class name or the instance name.

Class methods are methods that are defined inside a class and are called on the class itself rather than an instance of the class. Class methods take `cls` as the first parameter, which is a reference to the class itself. Class methods can access and modify class variables. To define a class method, you use the `@classmethod` decorator followed by the `def` keyword, the method name, and the `cls` parameter.

```python
class Person:
    count = 0

    def __init__(self, name, age):
        self.name = name
        self.age = age
        Person.count += 1

    @classmethod
    def get_count(cls):
        return cls.count

    @classmethod
    def create_person(cls, name, age):
        return cls(name, age)

print(Person.get_count())  # 0
person1 = Person.create_person("Alice", 30)
print(Person.get_count())  # 1
person2 = Person.create_person("Bob", 25)
print(Person.get_count())  # 2
```

In the example above, we defined a class method `get_count` that returns the value of the `count` class variable and a class method `create_person` that creates a new `Person` object. The `create_person` method is used to create new `Person` objects without directly calling the constructor.

### Static Methods

Static methods are methods that are defined inside a class but do not take `self` or `cls` as the first parameter. Static methods are used when the method does not depend on the instance or class and does not access or modify instance or class variables. To define a static method, you use the `@staticmethod` decorator followed by the `def` keyword and the method name. For example:

```python
class Math:
    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def subtract(a, b):
        return a - b

print(Math.add(3, 5))  # 8
print(Math.subtract(5, 3))  # 2
```

## Inheritance

Inheritance is a fundamental concept in object-oriented programming. Inheritance allows you to create a new class that inherits properties and behaviors from an existing class. The existing class is called the base class or parent class, and the new class is called the derived class or subclass. Inheritance allows you to reuse code and create a hierarchy of classes with shared properties and behaviors.

In Python, you can define a class that inherits from another class by specifying the base class in parentheses after the class name. For example:

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return "Animal sound!"

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

dog = Dog("Buddy")
cat = Cat("Whiskers")

print(dog.speak())  # Woof!
print(cat.speak())  # Meow!
```

### Base Classes and Derived Classes

Inheritance allows you to create a hierarchy of classes with shared properties and behaviors. For example, you can have a base class `Shape` with properties like `color` and methods like `area`, and derived classes like `Circle` and `Rectangle` that inherit from the `Shape` class and provide specific implementations for the `area` method.

```python
class Shape:
    def __init__(self, color):
        self.color = color

    def area(self):
        return 0

class Circle(Shape):
    def __init__(self, color, radius):
        super().__init__(color)
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, color, width, height):
        super().__init__(color)
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

circle = Circle("Red", 5)
rectangle = Rectangle("Blue", 3, 4)

print(circle.area())  # 78.5
print(circle.color)  # Red
print(rectangle.area())  # 12
print(rectangle.color)  # Blue
```

In the example above, we defined a base class `Shape` with a property `color` and a method `area`. We then defined two derived classes `Circle` and `Rectangle` that inherit from the `Shape` class and provide specific implementations for the `area` method. The `super()` function is used to call the constructor of the base class in the derived class.
