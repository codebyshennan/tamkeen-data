# Functions

## Learning Objectives

By the end of this lesson, you should be able to do the following:

- Define "control flow".
- Understand what a function is and why we use functions.
- Declare and define a function as a block of code.
- Understand how to pass input to a function.
- Understand how to store the output of a function.
- Understand the purpose of the return keyword.
- Execute a function.

## Introduction

{% include youtube.html id="VUcdDK1hSoE" %}

We can now start to group the code we're working with into "functions" and use functions to write more realistic, complex programs. Let's explore how functions work.

## What Functions Are

A function is a collection of lines of code, also known as a **"block"** of code. We run that code when we write the appropriate statement to "execute" the function. A function is a type of **"control flow"**, meaning function definitions in our code may not execute sequentially through the code file. For example, we may define a function at the top of our file, but only execute that function at the bottom of our file.

We can think of functions as "recipes" of code; a fixed set of instructions or steps, that can take in a variable input (e.g. number of dinner guests) and give a certain output (enough food for dinner), even though the instructions are statically defined. Functions have 3 key parts:

1. Input(s)
2. Pre-defined operations on the input
3. Output

Functions sometimes do not take in input, and sometimes do not return an output, but for now we will work with this basic structure.

![](https://cloudreports.net/wp-content/uploads/2019/06/Function_machine2.png)

## Why We Use Functions

Functions are necessary to abstract complexity away from the main logic of our applications. This allows us to break problems down into ever-smaller sub-problems, simplifying the process of coding complex applications. In this module we will see 2 use-cases of functions.

1. Purpose-built data manipulators that take inputs and give outputs. These are sometimes called _"helper"_ functions. We will define our functions' inputs and outputs, and all the instructions for that function will be contained in a block of code as part of the function definition.
2. Logic sequences that we wish to encapsulate in a "sub-routine". We do this with our `main` function. `main` contains everything we want to happen when the user clicks Submit. It is common for sub-routine functions to rely on several other helper functions.

## Define a Function

Define a function that "abstracts" the km-to-miles conversion we did in [2.3: Our First Program](../2-basic-data-manipulation/2.3-our-first-program.md#metric-conversion-program).

1. Start at the top of `script.js`
2. Create a variable for our function: `var convertKmToMiles`. **Note that function variable names start with a verb by convention, to distinguish them from variables storing other data types. With few exceptions such as the `main` function, function names should almost always start with a verb.**
3. Assign a function definition to that variable: `var convertKmToMiles = function () {}`


Note the beginning and end of the function "block" with curly braces (`{}`). We will add function logic within these curly braces. Semicolons marked the end of a line of code, curly braces are used to mark the start and end of a **code block.**


```javascript
var convertKmToMiles = function () {};
```

Define what logic happens when the function is executed. In our case, we want to convert KM to miles.

```javascript
var distanceInMiles = distanceInKm * 0.62;
```

Use the `return` keyword to define the function's output value. For our `convertKmToMiles` function, our return value is the distance in miles.


Note that the output of a function may not be the output of the overall program, because functions can call other functions. The output of the overall Coding Fundamentals starter code program is the return value of the `main` function.


```javascript
return distanceInMiles;
```

We define and name a function "input parameter" as `distanceInKm`. Functions can have 0 or more input parameters.

```javascript
var convertKmToMiles = function (distanceInKm) {
```

All together our function looks like the following.

```javascript
var convertKmToMiles = function (distanceInKm) {
  var distanceInMiles = distanceInKm * 0.62;
  return distanceInMiles;
};
```

### Functions Can Have Multiple Input Parameters

Functions can accept more than 1 input parameter, and input parameters are assigned based on position in the input parameter list, not by variable name.

For example, if we define a function with syntax like the following,

```javascript
var myFunc = function (myVar1, myVar2) { ... };
```

and we execute it with the following,

```javascript
// a and b are local variables
myFunc(a, b);
```

then inside of `myFunc`, `myVar1` will be assigned to `a` and `myVar2` will be assigned to `b`. The parameter assignment has nothing to do with the name of the input variables, and everything to do with the order of the variables passed into `myFunc`.

## Execute a Function

![](https://i.pinimg.com/236x/ef/9a/0f/ef9a0f91fb24fd8202fa7fe79253762e.jpg)

The syntax to run a function is to code the function name followed by parentheses (`()`). Without parentheses, the function is just a variable whose value is a function, and the function's logic will not run.

```javascript
convertKmToMiles();
```

Input parameters go within the function execution parentheses. Our `convertKmToMiles` function expects 1 input parameter, a distance in KM. For our function to work, we must pass this parameter to our function by specifying the parameter in the parentheses.

```javascript
var myKmDistance = 4;
convertKmToMiles(myKmDistance);
```

Let's test this function on its own outside of the Coding Fundamentals starter code.

1. Open Chrome Developer Tools by selecting View > Developer > JavaScript Console from the Chrome menu bar
2. Copy the function definition of `convertKmToMiles` into the console and hit `Enter`
3. The console should respond with an 'undefined', that is to be expected.
   - If the expression doesn’t return a value, the Chrome console will report undefined — this doesn’t mean anything is wrong, it’s just letting you know that the previous expression didn’t return a result.
4. Type the function name into the console and hit `Enter`. You should see the code block you defined.
5. Run the function from the console again with a different input value

```javascript
// Step 2 - copying the FUNCTION DEFINITION into the Chrome Dev Tools
var convertKmToMiles = function (distanceInKm) {
  var distanceInMiles = distanceInKm * 0.62;
  return distanceInMiles;
};

// Step 4 - checking the definition in Chrome Dev Tools
convertKmToMiles;

// Step 5 - executing the function using the FUNCTION CALL in the Chrome Dev Tools
convertKmToMiles(7676);

/*
 * REMEMBER that function call using the curved braces ()
 * is what 'activates' the function.
 *
 * Using the function without the curved braces is referencing
 * the variable name that stores the function (cf. step 4)
 */
```

### **Use Other Input Variables**

Create several variables and use them as input to your function.

```javascript
var kmsToHome = 4;
var kmsToSchool = 6;
var kmsToWork = 7;
```

Use these variables as input to our function. Paste each function call into the console and hit `Enter` one by one.

```javascript
convertKmToMiles(kmsToHome);
convertKmToMiles(kmsToSchool);
convertKmToMiles(kmsToWork);
```

## Capture the Result

We can see the result of our calculation in the console and capture that value for later. Paste this code into the console:

```javascript
var result = convertKmToMiles(7676);
```

Type in the name of the variable, `result`, and press `Enter` to see the given value.

## Add Function to Code File

Back in `script.js`, Call `convertKmToMiles` from our `main` function, providing `input` as a parameter to `convertKmToMiles`.

```javascript
var convertKmToMiles = function (distanceInKm) {
  var distanceInMiles = distanceInKm * 0.62;
  return distanceInMiles;
};

var main = function (input) {
  var myOutputValue = convertKmToMiles(input);
  return myOutputValue;
};
```

## Other Sample Functions

### Double a Number

```javascript
var doubleNumber = function (number) {
  return number * 2;
};
```

### Convert Kilograms to Pounds

```javascript
var convertKilosToPounds = function (kilos) {
  return kilos * 2.2;
};
```

### Calculate Area of a Circle

```javascript
var calcCircleArea = function (radius) {
  var pi = 3.1415;
  return pi * radius * radius;
};
```

## Main Function

We can now better understand the `main` function in our starter code. You may have noticed we currently do not execute `main` from `script.js`. Our starter code runs `main` for us whenever we click the Submit button. The code for this mechanism is inside `index.html`.

- The `input` parameter of `main` is what the user types in the input box
- The `return` value of `main` is what gets displayed in the output box

With this in mind, assign the value of `input` to `myOutputValue` to see this connection. Then change the code again to assign the string value "wow hello" to `myOutputValue`.

## Exercises (Base)

For each app, create a new copy of the Fundamentals Starter Code folder.

For each exercise try to encapsulate each distinct operation inside a function that returns a value. For example, `daysToMinutes(3)` is a function that converts days to minutes, takes a number of days as input, and returns a number of minutes.


Share your work with your section: save the code you write into another file. Name it `share.js` (A file only for sharing in the community channel.) Send the code file as a snippet in your section community channel.


### Run Example Code

1. Duplicate and run the code from this module
2. Paste the other function examples in instead of `convertKmToMiles`
3. Remove the `return` statements from the examples. What happens and why? What does the `return` statement do and how does it work?
4. Remove the `distanceInKm` parameter from the function definition. What happens and why? Replace the parameter afterward.
5. Change the variable name `distanceInKm` to another name everywhere. What happens and why? Change the name back to `distanceInKm` afterward.

### **Juice Wedding**

It takes 4 oranges to make a 90ml glass of orange juice. When planning your wedding, you need to know how many oranges to buy so all your guests have 1 glass of juice. The user will enter a number of guests and the app will say how many oranges are needed and how many litres of orange juice you would get. ([Ref Solution](https://github.com/rocketacademy/fundamentals-starter-code/tree/5.1-refSolns-juiceWedding))

### **SG Hugs**

Everyone has a different gauge for how long they like to hug. The user can enter a number of seconds they like to hug on average, and the app will calculate how many years it will take to hug everyone in Singapore. You are allowed 9 hours a day to sleep and eat. ([Ref Solution](https://github.com/rocketacademy/fundamentals-starter-code/tree/5.1-refSolns-sgHugs))

### **House Paint**

Estimate the price of painting the interior of your home. The user will enter a dollar amount of paint per litre and the app will calculate how much it will cost.

Each room in your 6-room house has the same area to be painted: 3m x 3m. There are 6 windows and 8 doors in your designer home. Coincidentally, windows and doors all of the same size: 90cm x 150cm. Paint will cover 300 square meters per litre. You want to apply 2 coats of paint. ([Ref Solution](https://github.com/rocketacademy/fundamentals-starter-code/tree/5.1-refSolns-housePaint))

### **Train Speed**

Two trains are leaving to Tokyo. Train 1 is traveling 200kph, and will reach Tokyo in 2 hours. Train 2 is newer and can travel faster, but is delayed due to a signalling fault. Build an app for Train 2's conductor calculate how fast Train 2 needs to travel to arrive at Tokyo at the same time as Train 1, based on how long it was delayed. Output the speed in kph. ([Ref Solution](https://github.com/rocketacademy/fundamentals-starter-code/tree/5.1-refSolns-trainSpeed))

## **Exercises (More Comfortable)**

### **Clock**

A user can enter the number of minutes past 1pm and the app will calculate the angle between the hour and minute hand. You are free to decide how else your clock will work: if the minute hand moves in 5 minute increments or moves every second, etc.
