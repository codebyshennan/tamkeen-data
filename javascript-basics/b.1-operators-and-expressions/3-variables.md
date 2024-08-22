# Assignment Operators | Variables

## Learning Objectives

By the end of this lesson, you should be able to:

- Describe what a variable is and how to assign it a value or expression.
- Use the `var` **keyword** to declare a new JavaScript variable.
- Explain how to achieve _accurate representation_ of your program with suitable variable names.
- Use the camelCase naming convention for your JavaScript variable names.

## Introduction

To do more than just basic math, we need to store, access and manipulate data in a computer's _memory_. We do this by utilizing **variables** in programming.

{% include youtube.html id="mBZrHgAja78" %}

## What are Variables?

Variables store and contain data. We name a variable and we associate a specific data value with it. We use variables to represent data that our program will process. First, we name a variable by using the **keyword** `var`. This tells the browser that the next word is the intended name for the variable.


Keywords are words that have a pre-defined meaning within a programming language, and are reserved for that use only. While all programming languages will have some way to declare a variable, different languages will do so using different keywords or syntax.


In order to assign a value to a variable, we use the assignment operator `=`. In programming, the equals sign has a slightly different meaning than in a mathematical equation, we are using `=` **declaratively**:

```javascript
var pi = 3.14;
var radius = 4;
```

In the above statements we have declared the variables `pi` and `radius`, and assigned them the value `3.14` and `4` respectively. Using these 2 data values we can calculate the area of a circle.

```javascript
pi * radius * radius;
```

We can also use variables to _capture_ the result of that calculated value, and store it in _memory_.

```javascript
var area = pi * radius * radius;
```

Note: You may have seen `let` and `const` syntax being used to declare variables in JavaScript. `let` and `const` are both _relatively_ new ways to declare variables in JavaScript, and while `let` has a similar use case to `var,` they are different in ways which we will not delve into right now. For the purposes of Coding Fundamentals, it will suffice to use `var`, but you are free to explore and experiment.

## **The Importance of Abstraction** <a href="#the-importance-of-abstraction" id="the-importance-of-abstraction"></a>

If we wanted the same result we could also write the following.1var x = 3.14 \* 4 \* 4;Copied!But this would not be as _meaningful_ as our previous example.Our programs must not only _calculate_ values correctly, they must also accurately _represent_ the operations we are performing, in this case calculating the area of a circle. **The names of our variables give meaning to the data our code contains.**

![Drake is not known for his programming skills. Don't be like Drake.](https://miro.medium.com/max/717/1*7LO7JEAZbo6YmN8feAVCwg.png)

## **Variable Naming Convention**

In JavaScript convention, ordinary variables are named with [lower camel case](<https://en.wikipedia.org/wiki/Naming_convention_(programming)#Examples_of_multiple-word_identifier_formats>). The name starts with a lowercase letter and every subsequent word starts with uppercase. For example, `bananaCount` or `firstDiceRoll`. This is _convention_, and will not affect code _functionality_. Different languages and communities have different conventions. Similar to driving conventions: in some countries, cars are driven on the left side, and on the right side in other countries.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/CamelCase_new.svg/1200px-CamelCase_new.svg.png)

## Exercises

Using the Chrome Dev Tools Console, write code to represent each of the following. Use descriptive variable names that explain what you are calculating. You will need to search the web for some formulas. When you are done, paste the code into a message to yourself in the community channel.

1. Calculate the circumference of a circle given its radius
2. Convert Celsius to Fahrenheit
3. Calculate the volume of a cube given a side length
4. Calculate the gradient percentage of a road given rise and run
5. Convert millilitres to pints
