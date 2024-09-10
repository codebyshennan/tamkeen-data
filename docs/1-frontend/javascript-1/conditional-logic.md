# Intro to Logic

## Learning Objectives

By the end of this lesson, you should be able to do the following:

* Understand how to control program inputs to quickly verify program logic
* Understand and use the `if` statement.
* Understand and use the equality operator: `==`
* Understand what logical operators are.
* Use the logical AND, OR, and NOT operators.

## Introduction

{% include youtube.html id="ywFgZIZXGoI" %}

So far our apps have always performed the _same operations_, no matter the input. The next level is to create programs that perform _different_ operations, depending on the input.

### What is Logic?

From a programming perspective, logic is the ability of the computer to **make decisions** based on input data.

To begin with logic we'll be using the most basic JavaScript logic syntax, the `if` condition. That means that some code will run or not depending on values we test. To start off with, those values will be what the user is typing in- so depending on what the user types, some different things will be displayed in the grey box.

## If Statement

An "**if statement**" is a control-flow "**code block**" that runs if a condition is `true`. A code block is a section of code surrounded by curly braces. We'll talk more about what `true` means when we introduce the boolean data type in next lesson.

### Simple Conditional Example: Secret Phrase

Let's write a program in our [starter code](https://github.com/SkillsUnion/fundamentals-starter-code) that changes the output value of `"hello world"` if we type in a particular phrase.

```javascript
  // Set a default value for myOutputValue
  var myOutputValue = 'hello world';
  var input = 'test phrase';
  // If input is our secret phrase, change the value of myOutputValue
  if (input == 'palatable papaya') {
    myOutputValue = 'you wrote the secret phrase!';
  }
  // return myOutputValue as output
  console.log(myOutputValue);
```

Our if statement starts after the declaration of the variable `input`. The conditional inside it tests if string stored in the variable `input` is equal to `'palatable papaya'`, our secret phrase. If it is equal to `'palatable papaya'`, the code runs between the curly braces, i.e. the "if block". If it is not equal to our phrase, the if block does not run.


Try changing the value of the `input` variable and non-secret phrases into the program and see how the if-statement works.

### Equality

We're using the equality operator `==` to test if `input` is equal to `'palatable papaya'`. There are two comparison operators in JavaScript to check for equality, `==`, known as the abstract equality operator, and `===`, the strict equality operator.


There are other "[**comparison operators**](https://www.w3schools.com/js/js_comparisons.asp)" in this documentation, feel free to explore and try things out.


### if-else statements

There will be times that there is a need for a code block to run when the if-statement's condition is false. This is where the `else` keyword comes in. The `else` code block follows the `if` code block and it executes when the `if` code block doesn't.

```javascript
  var myOutputValue = 'hello world';
  var input = 'test phrase';
  if (input == 'palatable papaya') {
    myOutputValue = 'you wrote the secret phrase!';
  } else { //This would run and replace the content of myOutputValue
    myOutputValue = 'you have entered the wrong phrase';
  }
  console.log(myOutputValue);
```

The `else` code block runs because the comparison between `input` and the phrase is false. This allows the value of `myOutputValue` change depending on what the input was.

## Knowledge Application: Dice Game

In order for us to understand more how logic works, let's build a simple dice game application. Let's start by building a **function** that generates random dice numbers. We would learn more about functions in the next lessons.

We will be using this 'Dice Rolling function' as a base to explore Logic and Control Flow for the rest of this lesson.

### Random Number Generation

To simulate dice, we first need random number generation. JavaScript can produce random numbers using a built-in "**library**" called [`Math`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math) (case-sensitive). `Math` contains functions that perform common and helpful math operations.

The function we need is `Math.random()`. Calling `Math.random()` returns a random decimal number between 0 and 1, inclusive of 0 and exclusive of 1.

```javascript
var myRandomValue = Math.random();
```

Since we wish to simulate dice with numbers between 1 to 6 inclusive, we have to manipulate the randomly-generated number to get what we want.

To convert our random number to a valid dice roll value, we'll use another `Math` function: `Math.floor()`. We will follow the random integer generation example [here](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/random) to use `Math.floor()` to convert decimals to integers.

With `Math.random()` and `Math.floor()`, we can make a function that produces any random integer from 0 to a provided `max` number:

```javascript
var getRandomInteger = function (max) {
  // Generate a decimal from 0 through max + 1.This decimal will be inclusive of 0 and exclusive of max + 1.
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

Using our new knowledge of '**logic**' in this chapter, let's implement a simple rule to create a "dice game": *if the user enters the same number as the dice roll, they win.*

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

Try to make the program work before moving to the next section.


## Boolean Operators

Boolean operators are operators that return boolean values. Boolean operators consist of both logical operators (e.g. `||`, `&&`, `!`) and comparison operators (e.g. `==`, `<`, `>`). Boolean values are either `true`, or `false`. Boolean operators, similar to math operators (e.g. `+`, `-`, `*`, `/`), always return a new value that can be stored in a variable.

### Logical OR

{% include youtube.html id="3QgxbnIOYnw" %}

Let's update our dice game from the previous section so that the user also wins if they guess within 1 of the dice roll. To encode this we will need a new kind of conditional syntax: "**logical operators**". 

The logical OR operator allows us to combine 2 [boolean expressions](https://en.wikipedia.org/wiki/Boolean_expression) into a single boolean expression. It will pass it as `true` if either one of the expression is `true`.

The JavaScript syntax for logical OR is `||`. The final syntax for our updated dice game logic might look like the following.

```javascript
if (
  randomDiceNumber == input ||
  randomDiceNumber + 1 == input ||
  randomDiceNumber - 1 == input
) {
  myOutputValue = 'you win';
}
```

### Boolean AND, NOT

We will add another dice to our dice game, and modify our logic such that the player only wins if both dice rolls are the same value as the player's guess. We will utilize the AND and NOT boolean operators for this task.

### AND Operator

Use the boolean AND operator (`&&`) to validate if 2 boolean statements are both true.

**JavaScript**

```javascript
var main = function (input) {
  var randomDiceNumber1 = rollDice();
  var randomDiceNumber2 = rollDice();
  // The default output value is "you lose".
  var myOutputValue = 'you lose';
  // If the input matches both random dice numbers, output value is "you win".
  if (randomDiceNumber1 == input && randomDiceNumber2 == input) {
    myOutputValue = 'you win';
  }
  return myOutputValue;
};
```

### Dice Game with 2 Dice and Snake Eyes

{% include youtube.html id="Aelo-Ay71oA" %}

Let's add another rule to our 2-dice dice game: the player loses if the 2 dice roll "snake eyes": 2 1s.

### NOT Operator

**Non-equivalence Example**

Paired with `=`, `!=` is a "not equals" comparison operator, contrasting `==`. The following example runs line 2 if the value of `diceRoll` is not 2.

```javascript
if (diceRoll != 2) {
  console.log('dice roll is not 2!');
}
```

**Reverse-Boolean Example**

The boolean NOT operator (`!`) is used to "reverse" a boolean statement. The following code examples show a conditional that checks if the player did not win.

When comparing against `true` or `false` values, we might tend to write `didPlayerWin != true` in our conditional.

```javascript
var didPlayerWin = false;

if (didPlayerWin != true) {
  console.log('player didnt win!');
}
```

However, the best practice is to put the `!` operator in front of the boolean variable, like `!didPlayerWin`. The following example is semantically the same as the previous example.

```javascript
var didPlayerWin = false;

if (!didPlayerWin) {
  console.log('player didnt win!');
}
```

### Encode Snake Eyes with NOT Operator

**Snake Eyes Boolean Statement**

The following statement encodes snake eyes in our program, where both `diceRoll1` and `diceRoll2` have a value of 1.

```javascript
diceRoll1 == 1 && diceRoll2 == 1;
```

**NOT Snake Eyes Boolean Statement**

To express NOT snake eyes, we can negate the boolean statement above with the `!` operator, making sure to wrap the snake eyes statement with parentheses (`()`) to indicate that the `!` applies to the entire snake eyes statement and not just the first part. The following evaluates to `true` when the dice rolls do NOT represent snake eyes, i.e. when `diceRoll1` and `diceRoll2` do not both have the value of 1.

```javascript
!(diceRoll1 == 1 && diceRoll2 == 1);
```

The final code should look similar to this:

```javascript
var main = function (input) {
  var randomDiceNumber1 = rollDice();
  var randomDiceNumber2 = rollDice();

  var myOutputValue = 'you lose';

  if (randomDiceNumber1 == input && randomDiceNumber2 == input) {
    myOutputValue = 'you win';
  }

  if (!(diceRoll1 == 1 && diceRoll2 == 1)){
    myOutputValue = 'you win';
  }

  return myOutputValue;
};
```

## Exercise

### **Easier Dice Game**

Change the dice game so that it's easier. If the user guess is within 2 of the dice roll, they still win.

## Further Reading

Past students have found this [slide deck](https://www.cs.cmu.edu/~mrmiller/15-110/Handouts/boolean.pdf) helpful in understanding Boolean logic.
