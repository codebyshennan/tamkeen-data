# Intro to Logic

## Learning Objectives

By the end of this lesson, you should be able to do the following:

* Know how logic is used to make programs more complex
* Understand and use the `if` statement.
* Understand and use the equality operator: `==`

### Pseudo-Code, Boolean Or

* Explain why writing pseudo-code can be helpful, and know how to write simple psuedo-code.
* Understand what logical operators are.
* Understand and use the logical OR operator.

### Boolean AND, NOT

1. Code control: how to control program inputs to quickly verify program logic.
2. Boolean values: what are boolean values and how are they stored in JavaScript?
3. Boolean AND operator
4. Boolean NOT operator
5. Conditional debugging: how to efficiently debug conditional logic.



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


# Pseudo-Code, Boolean Or

## Learning Objectives

By the end of this lesson, you should be able to:

* Explain why writing pseudo-code can be helpful, and know how to write simple psuedo-code.
* Understand what logical operators are.
* Understand and use the logical OR operator.

## Introduction

{% include youtube.html id="3QgxbnIOYnw" %}

Let's update our dice game from the previous lesson so that the user also wins if they guess within 1 of the dice roll. To encode this we will need a new kind of conditional syntax: "**logical operators**". We will also learn about "**pseudo-code**", a common method to plan logic before writing code.

## Pseudo-Code: Logical Expression Translation

As our code gets more complex, we will often want to plan our logic before writing code. This helps clarify our thoughts such that our code is more organised and we spend less time debugging.

### Example: Dice Game Pseudo-Code

Let's plan our logic for the updated dice game. A sample plan might look like this.

```
If the guess is correct, the user wins.
If the guess is off by 1, the user wins.
```

To start translating this to code, we can clarify the "off by 1" statement by breaking it down into the following 2 statements. We can also revise the "correct" statement using similar language.

```
If the guess plus one equals the dice roll, the user wins.
If the guess minus one equals the dice roll, the user wins.
```

Altogether we would have the following.

```
If the guess equals the dice roll, the user wins.
If the guess plus one equals the dice roll, the user wins.
If the guess minus one equals the dice roll, the user wins.
```

Notice how we reformulated sentences describing game logic at a "higher level" into a "lower-level" format that more closely resembles code. This step can be helpful when we are unsure how to translate higher-level logic descriptions to code.

#### Pseudo-Code Sentence

```
If the guess equals the dice roll, the user wins.
```

#### Corresponding Code Snippet

```javascript
if (input == randomDiceNumber) {
  myOutputValue = 'you win';
}
```

### The Importance of Pseudo-Code

Pseudo-coding is an important intermediate step to translate program requirements to code, helping us minimise logical errors. Before coding any program, try to translate program requirements into plain-English logical statements in code-comment format, then fill in your code beneath each line of pseudo-code.

Almost all developers pseudo-code, regardless of seniority. Even when we may not need pseudo-code to develop a program correctly, others reading our code will find it helpful to read the pseudo-code.

## Logical Operators

### Code Without Logical Operators

From our pseudo-code above we obtained the following specification.

```
If the guess equals the dice roll, the user wins.
If the guess plus one equals the dice roll, the user wins.
If the guess minus one equals the dice roll, the user wins.
```

We could translate this to a series of **if-statements**.

```javascript
if (randomDiceNumber == input) {
  myOutputValue = 'you win';
}

if (randomDiceNumber + 1 == input) {
  myOutputValue = 'you win';
}

if (randomDiceNumber - 1 == input) {
  myOutputValue = 'you win';
}
```

To write more concise code, we'll introduce another conditional syntax: logical operators. Logical operators allow us to construct more complex logical statements.

### Logical OR: Any of These Statements

The logical OR operator allows us to combine 2 [boolean expressions](https://en.wikipedia.org/wiki/Boolean_expression) into a single boolean expression. A boolean expression is an expression that evaluates to either `true` or `false`. To combine more than 2 boolean expressions in a single statement, we can use multiple OR operators.

Here is pseudo-code that more closely reflects OR logic.

```
If any of the following are true, the user wins.

- The guess equals the dice roll
- The guess plus one equals the dice roll
- The guess minus one equals the dice roll
```

We could also break our pseudo-code down more explicitly. This latter pseudo-code most closely resembles how we will construct our code.

```
If:

The guess equals the dice roll,

OR

The guess plus one equals the dice roll,

OR

The guess minus one equals the dice roll,

the user wins.
```

### Logical OR Syntax

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

In next lesson, we will learn about 2 more logical operators, AND (`&&`) and NOT (`!`).

## Exercises (Base)

### Follow Along

Duplicate and run the code above.

### **Secret Words**

Change the original [secret word program in section 4.1](https://fundamentals.rocketacademy.co/4-conditional-logic/4.1-intro-to-logic#simple-conditional-example-secret-phrase) so that the user could enter more than one secret word- they can enter "neat noodles", "awesome ayam", "delicious dumplings" or the original word to see the secret message.

### **Easier Dice Game**

Change the dice game so that it's even easier. If the user guess is within 2 of the dice roll, they still win.

### **Even Easier Dice Game**

The user can guess by putting in one of two values: "odd" or "even". They win if the dice roll is odd or even.

## Exercises (More Comfortable)

### **Combo Game**

Change your dice game so that if the user types in "palatable papaya" instead of a dice guess, they also win.

### **Even Easier Dice Game Refactor**

There is a way to mathematically verify that a number is even using the `%` operator - the modulus operator. Google the solution to tell if a number is even in JavaScript, and what the modulus operator does and refactor the logic in the game to use it.


# Boolean AND, NOT

## Introduction

In previous lesson, we learned how to pseudo-code and use logical OR operators. In this module we will learn about the following.

1. Code control: how to control program inputs to quickly verify program logic.
2. Boolean values: what are boolean values and how are they stored in JavaScript?
3. Boolean AND operator
4. Boolean NOT operator
5. Conditional debugging: how to efficiently debug conditional logic.

## Code Control

{% include youtube.html id="AYmtC4nWKsU" %}

Sometimes we will want to control program inputs to verify program logic for specific inputs. This is especially relevant when we would otherwise not be able to control these inputs, for example when using a random number generator. Even though the user did not input the randomly-generated number, the number can be considered an input because it determines logic within the program.

No matter how small or large the range of randomness in our input, we want to be able to control this randomness during development to test our logic for potential inputs. In other words, we don't want to continuously enter numbers and click submit in our browsers until we reach a win state. If our dice rolls involved 100-sided dice, our hands would get tired.

We will control random number input to test winning conditions by removing randomness. We will "fix" the random number by changing `rollDice` to always return a specific value. In the following code snippet, `rollDice` always returns 6. Note that the `return` keyword also ends the execution of the function: `rollDice` never runs beyond line 2. We will remove `return 6;` once we're done testing.

```javascript
var rollDice = function () {
  return 6;
  // produces a float between 0 and 7
  var randomFloat = Math.random() * 7;
  // take off the decimal
  var resultInteger = Math.floor(randomFloat);
  return resultInteger;
};
```

## Boolean Values

Boolean operators are operators that return boolean values. Boolean operators consist of both logical operators (e.g. `||`, `&&`, `!`) and comparison operators (e.g. `==`, `<`, `>`). Boolean values are either `true`, or `false`. Boolean operators, similar to math operators (e.g. `+`, `-`, `*`, `/`), always return a new value that can be stored in a variable.

What is the value of `myVal` in each of the following examples?

**Math Operator Example**

```javascript
// the value of myVal is the result of the number 1 PLUS the number 2.
var myVal = 1 + 2;
```

**Boolean Operator Example**

```javascript
// the value of myVal is the result of the number 1 BOOLEAN EQUALS the number 2
var myVal = 1 == 2;
```

In previous modules we've seen number and string data types. Boolean data types are a 3rd data type. They represent a value that is `true` or `false`. Just like how variables can store the result of a math operation, variables can also hold the result of a logical, boolean operation.

**Boolean Variable Example**

Note the naming convention for variables that store booleans is to begin with a question word to signal that the variable's value is either `true` or `false`.

```javascript
// Assign true to the didUserWin variable
var didPlayerWin = true;
```

**Boolean Operation and Assignment Example**

Assign the result of `input == randomDiceNumber` to `didPlayerWin`. The result will be `true` or `false`. Notice this matches our dice game logic, which says that the player wins if their input matches the random dice number.

```javascript
var didPlayerWin = input == randomDiceNumber;
```

## Dice Game Example

We will add another dice to our dice game, and update our logic such that the player only wins if both dice rolls are the same value as the player's guess. We will utilize the AND and NOT boolean operators for this task.

### AND Operator

Use the boolean AND operator (`&&`) to validate if 2 boolean statements are both true.

**Pseudo Code**

```
if
the guess is equal to the first random number
AND
the guess is equal to the second random number
then the user wins.
```

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

Use the boolean NOT operator (`!`) to "reverse" a boolean statement. The following code examples show a conditional that checks if the player did not win.

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

**Dice Game with 2 Dice and Snake Eyes Pseudo Code**

```
if
the guess is equal to the first random number
AND
the guess is equal to the second random number
AND
the dice are NOT snake eyes
then the user wins.
```

**Snake Eyes Boolean Statement**

The following statement encodes snake eyes in our program, where both `diceRoll1` and `diceRoll2` have a value of 1.

```javascript
diceRoll1 == 1 && diceRoll2 == 1;
```

**NOT Snake Eyes Boolean Statement**

To express NOT snake eyes, we can negate the above boolean statement with the `!` operator, making sure to surround the snake eyes statement with parentheses (`()`) to indicate that the `!` applies to the entire snake eyes statement and not just the first part. The following evaluates to `true` when the dice rolls do NOT represent snake eyes, i.e. when `diceRoll1` and `diceRoll2` do not both have the value of 1.

```javascript
!(diceRoll1 == 1 && diceRoll2 == 1);
```

## Conditional Debugging

{% include youtube.html id="yXtC954S2W4" %}

We've written pseudo-code, we've written JavaScript, we've controlled and tested our logic, yet our code could still have errors. Let's analyse common conditional errors and learn strategies to diagnose them.

### Consider Common Conditional Errors

#### Boolean Expression Not Evaluating to True

When we use `&&` , both boolean statements must be true for the combined statement to be true. If we only need 1 of the boolean statements to be true, use `||` instead.

#### Boolean Expression has Syntax Error

Did we unintentionally use the assignment operator `=` instead of the comparison operator `==`?

#### Incorrect Logic in Boolean Statement

Are the parentheses in our boolean statement arranged such that the boolean logic matches what we wish to accomplish?

### Deconstruct and Verify Each Condition

If the above common conditional errors are insufficient to debug our problem, we can use a more fine-grained debugging strategy. By now, many of our conditionals comprise multiple boolean statements connected by logical operators such as `||`, `&&`, and `!`. If the program behaviour is incorrect, we can methodically deconstruct our conditionals to verify each component.

For a given conditional like the following, we can console log each component of the conditional. We can then inspect each variable and boolean expression value. Note the helpfulness of labelling each console log for output clarity. Try not to introduce bugs while debugging by mislabeling console logs.

#### Sample Conditional

```javascript
if (
  randomDiceNumber == input ||
  randomDiceNumber + 1 == input ||
  randomDiceNumber - 1 == input
) {
  myOutputValue = 'you win';
}
```

#### Sample Debugging Code

```javascript
console.log('random dice number:');
console.log(randomDiceNumber);
console.log('input');
console.log(input);
console.log('random dice equals input:');
console.log(randomDiceNumber == input);
console.log('random dice plus 1 equals input:');
console.log(randomDiceNumber + 1 == input);
console.log('random dice minus 1 equals input:');
console.log(randomDiceNumber - 1 == input);
```

## Exercises


Share your work with your section: save the code you write into another file. Name it `share.js` (A file only for sharing in the community channel.) Send the code file as a snippet in your section community channel.


### Follow Along

Duplicate and run the dice game versions above.

### New Winning Conditions

Create new versions of our dice game for each of the following winning conditions.

1. User wins if guess is within 1 for any of 2 dice.
2. User wins if guess is within 1 for all 2 dice.
3. User wins if guess is within 1 of either dice but the user does not roll snake eyes.
4. User wins if guess is within 1 of either dice or if the user rolls snake eyes.


If the above exercises are taking a long time, we may wish to complete the subsequent pre-class modules before returning to these exercises.


## Further Reading

Past students have found this [slide deck](https://www.cs.cmu.edu/~mrmiller/15-110/Handouts/boolean.pdf) helpful in understanding Boolean logic.
