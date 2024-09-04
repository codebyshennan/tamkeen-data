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
