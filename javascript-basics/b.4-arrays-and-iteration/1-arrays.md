# Arrays

## Learning Objectives

By the end of this lesson, you should be able to:

* Explain what an array is.
* Use arrays to store similar "kinds" of data.
* Access data stored in an array.
* Use the built-in array methods of `push()` and `pop()` to manipulate data in an array

## Introduction

{% include youtube.html id="CI09j_aHLE0" %}

We've built relatively complex programs using functions and conditionals with global state. In this module we will learn a new data type, or "data structure", to be precise: dynamic-length data "**arrays"**. We'll then learn how to manipulate arrays using a syntax called "**loops**". Just like each concept we've learned previously, the mechanic of the individual concepts will be relatively simple, but when we combine these concepts with previous concepts, the complexity of our apps can increase greatly.

## Arrays are a Type of Data Collection

Values of each data type we have seen so far can be stored inside variables, e.g. numbers, strings, booleans, and even functions. Arrays can also be stored inside variables, but have an additional capability: to store 0 or more variables inside them. Arrays can thus be described as variable-length **"data collections"**, because they are a "collection" of data.

## Arrays Only Store 1 "Kind" of Data at a Time

Arrays are data structures that store 0 or more values of the same kind. By "kind" we don't just mean the same data type, e.g. numbers, strings, or booleans. By "kind" we also mean the same conceptual kind of data. For example, we might store computer-generated dice rolls in an array, or user guesses in an array, where both dice rolls and user guesses are numbers, but we would likely not store both dice rolls and user guesses in the same array, because each is a different "kind" of data.


JavaScript lets us add any data type into any array, regardless of what's already inside. The following would be considered valid JavaScript syntax.

```javascript
var myValues = [23, 'rocks', true, 0.345, 'awesome'];
```

However, the **concept** of the array is for holding data of the same kind, e.g. numerical dice rolls, and this almost always means the data is of the same JavaScript data type. Please do not deviate from this in SWE Fundamentals.


## 2 Conceptual Types of Arrays

We can use arrays in 2 conceptual ways: to store static-length data and to store dynamic-length data.

### Arrays as Static Data Collections

1 way we can use arrays is to store data that will never change. Examples of this are names of days of the week, and letters in the English alphabet. After we initialise these arrays, we will most likely only read from them, and not edit the values inside the arrays.

#### Example: Array Containing Names of Days of the Week

```javascript
var daysOfTheWeek = [
  'Monday',
  'Tuesday',
  'Wednesday',
  'Thursday',
  'Friday',
  'Saturday',
  'Sunday',
];
```

**Example: Array Containing Letters in the Alphabet**

```javascript
var letters = ['a', 'b', 'c'];
```

### Arrays as Static-Length Collections with Dynamic Data

Another way we can use arrays is to store data that may change, but will always have the same number of elements. An example of this is storing temperatures of each day in the past week. There are always 7 days in the past week, thus there will always be 7 elements in our array, but the value of each element can change depending on the temperature on the relevant day.

**Example: Temperatures of Each Day in Past Week**

```javascript
var dailyAverageTemperatureInPastWeek = [30, 32, 29, 34, 36, 38, 32];
```

### Arrays as Dynamic Data Collections

We can represent data that our program needs to alter as it runs. The program is responsible for keeping track of these data values, and making the array grow larger or smaller.

**Record a list of all guesses a user makes in the game.**

```javascript
var winningDiceGuesses = [3, 4, 2, 1, 4];
```

#### List of the sales recorded up to the current day (whichever current date this program runs)

```javascript
var salesThisMonth = [34, 23, 12, 78, 56];
```

## Array Properties

Arrays have unique properties that allow us to represent data in new ways: **Position** and **order**.

### Position

We access elements in an array through **"indexes"** that represent the position of each array element. In the following example, element 1 is at index 0, element 2 at index 1, and element 3 at index 2. Arrays are 0-indexed, meaning the first element is always at index 0. Index 0 is always the left-most array position, and the last index is the right-most array position.

```
  0    1    2
['a', 'b', 'c']
```

We can use square-bracket (`[]`) syntax to retrieve any element in an array by specifying that element's index in the square brackets.

```javascript
var letters = ['a', 'b', 'c'];
var value = letters[2];
```

What happens when we try to access an array element that doesn't exist?

```javascript
letters[234534534534];
```


#### Using User Input as Array Index

The following example uses user input to retrieve a value at a specific index in the `letters` array.

```javascript
var letters = ['a', 'b', 'c'];

var main = function (input) {
  var myOutputValue = 'your letter: ' + letters[input];
  return myOutputValue;
};
```

### Order

Array positions imply order, e.g. the element at index 0 comes before the element at index 1. The following array implies "a" is before "b".

```javascript
var letters = ['a', 'b', 'c'];
```

## Array Data Manipulation

Let's learn how to alter, insert, and remove data from arrays. Notice the following code does not manipulate array values. It stores the value of the 3rd array element in a variable `value`, but does not alter the `letters` array.

```javascript
var letters = ['a', 'b', 'c'];
var value = letters[2];
```

The following are methods to alter values in arrays.

### Altering Existing Values in Arrays

Create an array with values inside.

```javascript
var temperatures = [23, 43];
```

Alter a value inside the array by assigning a new value to the original value's location in that array. This will only alter the value at the specified location in the array, and not any other values. This allows us to reassign values in the array after we initialise the array.

```javascript
// Reassign the value at index 0 to a new value.
temperatures[0] = 99;
```

### Inserting and Removing New Elements from Arrays

There are 2 methods we will use often in SWE Fundamentals to add and remove elements from arrays: `push` and `pop`.

#### `push`

`push` adds a value to the **end** of an existing array. `push` accepts the value to be inserted as a parameter.

```javascript
var temperatures = [23, 43];
// Add the parameter to push (33) to the end of the temperatures array.
// The temperatures array will be [23, 43, 33].
temperatures.push(33);
```

#### **`pop`**

`pop` removes the **last** element from an array and returns it.

```javascript
var temperatures = [23, 43];
// Remove the last element (43) from the temperatures array.
// The temperatures array will be [23], and lastElement will be 43.
var lastElement = temperatures.pop();
```

### Example: Add User Input to Array

The following is an example dice game that records and outputs all prior user guesses.

```javascript
// Store guesses in a global array that persists across Submits.
var guesses = [];
var main = function (input) {
  // Add the user's guess to the guesses array.
  guesses.push(input);
  // Generate a random dice number.
  var randomDiceNumber = rollDice();
  // Initialise output to communicate loss.
  // Output the record of all guesses regardless of loss or win.
  var myOutputValue = 'You lose. Your guesses: ' + guesses;
  // If the guess matches the dice roll, change output to communicate win.
  if (randomDiceNumber == input) {
    myOutputValue = 'You win. Your guesses: ' + guesses;
  }
  // Return output value.
  return myOutputValue;
};
```

## Exercises

### Follow Along

Implement the above code and verify results.
