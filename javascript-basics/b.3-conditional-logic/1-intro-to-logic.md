# Intro to Logic

## Learning Objectives

By the end of this lesson, you should be able to do the following:

* Know how logic is used to make programs more complex
* Understand and use the `if` statement.
* Understand and use the equality operator: `==`

## Introduction

{% include youtube.html id="ywFgZIZXGoI" %}

So far our apps have always performed the _same operations_, no matter the input. The next level of complexity is to create programs that perform _different_ operations, depending on the input.

### What is Logic?

From a programming perspective, logic is the ability of the computer to **make decisions** based on input data.

To begin with logic we'll be using the most basic JavaScript logic syntax, the `if` condition. That means that some code will run or not depending on values we test. To start off with, those values will be what the user is typing in- so depending on what the user types, some different things will be displayed in the grey box.

## If Statement

An "**if statement**" is a control-flow "**code block**" that runs if a condition is `true`. A code block is a section of code surrounded by curly braces. We'll talk more about what `true` means when we introduce the boolean data type in next lesson.

### Simple Conditional Example: Secret Phrase

Let's write a program that changes the output value of `"hello world"` if we type in a particular phrase.

```javascript
var main = function (input) {
  // Set a default value for myOutputValue
  var myOutputValue = 'hello world';
  // If input is our secret phrase, change the value of myOutputValue
  if (input == 'palatable papaya') {
    myOutputValue = 'you wrote the secret phrase!';
  }
  // return myOutputValue as output
  return myOutputValue;
};
```

Our if statement is on line 5. The conditional inside it tests if `input` is equal to `'palatable papaya'`, our secret phrase. If `input` is equal to `'palatable papaya'`, the code runs between the curly braces on lines 5 and 7, i.e. the "if block". If `input` is not equal to our phrase, the if block does not run.

Code blocks may or may not run depending on "[**control flow**](https://en.wikipedia.org/wiki/Control_flow)", i.e. the logic of our app. The 1st way we learned to use code blocks was with functions. If statements are the second way. We'll learn a 3rd code block syntax later called loops.

Try inputting secret and non-secret phrases into the program. Enter the secret phrase and click the button to see the different output. Enter anything else and click the button to see the default output.


Note the distinction in our code example between variable **declaration** and **assignment**. On line 2 we "declare" the variable `myOutputValue`. This creates the named container that we can store values inside. On line 5 we "assign" a new value to the `myOutputValue` container. The old value is overwritten and non-retrievable. Notice that declaration with the `var` keyword is only done once per variable. Please do not use `var` when assigning new values to existing variables in conditionals like if statements.


### Equality

We're using the "[**comparison operator**](https://www.w3schools.com/js/js_comparisons.asp)" `==` to test if `input` is equal to `'palatable papaya'`.


There are two comparison operators in JavaScript to check for equality, `==`, known as the abstract equality operator, and `===`, the strict equality operator. For the purpose of this course, `==` will suffice, but you are free to explore and experiment. You can read more about the in-depth differences between them in [this](https://stackoverflow.com/questions/359494/which-equals-operator-vs-should-be-used-in-javascript-comparisons) discussion.

## Comments

As our apps get more complicated, we can and should leave notes to ourselves and others to clarify what our code does. "**Comments**" let us write notes in our code files that are ignored on program execution. In JavaScript, comments are denoted by 2 slashes (`//`) at the start of the comment. Every programming language has commenting functionality, though commenting syntax varies by language.

```javascript
// This is an example comment. It won't actually "run".
```

## Knowledge Application: Dice Game

Let's recap and build onto functions lesson by building a **function** that generates random dice numbers.

We will be using this 'Dice Rolling function' as a base to explore Logic and Control Flow for the rest of this Module.

### Random Number Generation

To simulate dice, we first need random number generation. JavaScript can produce random numbers using a built-in "**library**" called [`Math`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math) (case-sensitive). `Math` contains functions that perform common and helpful math operations.

The function we need is `Math.random()`. **Note the function call using ()**

```javascript
var myRandomValue = Math.random();
```

Calling `Math.random()` returns a random decimal number between 0 and 1, inclusive of 0 and exclusive of 1.

Note that the `Math.random()` function does not take in an input.

Since we wish to simulate dice with numbers between 1 to 6 inclusive, we have to manipulate the randomly-generated number to get what we want.

To convert our random number to a valid dice roll value, we'll use another `Math` function: `Math.floor()`. We will follow the random integer generation example [here](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/random) to use `Math.floor()` to convert decimals to integers.

With `Math.random()` and `Math.floor()`, we can make a function that produces any random integer from 0 to a provided `max` number:

```javascript
var getRandomInteger = function (max) {
  // Generate a decimal from 0 through max + 1.
  // This decimal will be inclusive of 0 and exclusive of max + 1.
  var randomDecimal = Math.random() * (max + 1);

  // Remove the decimal with the floor operation.
  // The resulting integer will be from 0 through max, inclusive of 0 and max.
  var resultInteger = Math.floor(randomDecimal);

  return resultInteger;
};
```

### Dice Roll Program Logic

Since we wish to build a program that generates random dice numbers. We use the logic from the above `getRandomInteger` to always return an integer from 1 to 6. We call the new function `rollDice`.

```javascript
var rollDice = function () {
  // Generate a decimal from 0 through 6, inclusive of 0 and exclusive of 6.
  var randomDecimal = Math.random() * 6;

  // Remove the decimal with the floor operation.
  // This will be an integer from 0 to 5 inclusive.
  var randomInteger = Math.floor(randomDecimal);

  // Add 1 to get valid dice rolls of 1 through 6 inclusive.
  var diceNumber = randomInteger + 1;

  return diceNumber;
};
```

### Dice Game Logic

Using our new knowledge of '**logic**' in this chapter, let's implement a simple rule to create a "dice game": <mark style="background-color:blue;">if the user enters the same number as the dice roll, they win.</mark>

```javascript
var main = function (input) {
  // Generate a random dice number
  var randomDiceNumber = rollDice();

  // Default output value is 'you lose'.
  var myOutputValue = 'you lose';

  // If input matches randomDiceNumber, update output.
  if (input == randomDiceNumber) {
    myOutputValue = 'you win';
  }

  // Return output.
  return myOutputValue;
};
```

## Exercises

### **Follow Along**

Duplicate and run the code above.

### **Twice the Guess**

Update our dice game logic such that the user wins if the dice roll is 2 times the guess, e.g. a guess of 1 and roll of 2, a guess of 2 and roll of 4, etc. To win this updated game for a 6-sided dice, the user should only guess numbers between 1 and 3, but the game does not restrict what the user can guess.
