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
