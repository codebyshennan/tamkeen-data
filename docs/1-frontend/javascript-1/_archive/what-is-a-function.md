# What is a function?

A function is just a collection of lines of code. We can run those lines of code when we write the appropriate code statement.

A function is a type of **control flow**, which means we can write code that **does not** just execute line by line downwards through the code file. We can control the order and flow of the execution of our lines of code.

### Define a function.

First we'll define a function that isolates the kilometers to miles conversion we were making before, then we'll use it in our program.

Start at the bottom of our `script.js` file.

Create a variable `var kilometersToMiles`

We assign a function to that variable: `var kilometersToMiles = function`

We note the beggining and end of the function `block` with **curly braces**:

```js
var kilometersToMiles = function () {};
```

We define what actions happen when the function will be executed:

```js
var distanceInMiles = distanceInKilometers * 0.62;
```

Using `return` keyword we define the output value of the function:

```js
return distanceInMiles;
```

We define the function input called a `parameter` as `distanceInKilometers`.

```js
var kilometersToMiles = function (distanceInKilometers) {};
```

All together it looks like this:

```js
var kilometersToMiles = function (distanceInKilometers) {
  var distanceInMiles = distanceInKilometers * 0.62;
  return distanceInMiles;
};
```

### Execute a function

Let's run this function without the rest of our starter code first.

The syntax for running our function is to mention the name of the function and add **parentheses** to the end of it:

```js
kilometersToMiles();
```

Our function takes an argument - an actual value representing kilometers:

```js
var someNumber = 4;
kilometersToMiles(someNumber);
```

Let's test this without having to worry about the rest of the starter code.

Open your developer tools by going to the top level menu and selecting view -&gt; developer -&gt; javascript console.

Paste the above code into the console.

You should see the calculated result.

Type the name of the function into the console. You should see the code `block` you defined.

You can run this code again from the console with a different value:

```js
kilometersToMiles(7676);
```

Type in your own values.

### capture the result

We can see the result of the calculation in the console.

We can also capture that value to use it later.

```js
var result = kilometersToMiles(7676);
```

Type in the name of the variable: `result` and press enter to see the given value.

### Putting it all together

Inside main call the functions and give the input value as an argument.

```js
var main = function (input) {
  var myOutputValue = kilometersToMiles(input);
  return myOutputValue;
};

var kilometersToMiles = function (distanceInKilometers) {
  var distanceInMiles = distanceInKilometers * 0.62;
  return distanceInMiles;
};
```

## other examples

### Doubling a number

```js
var double = function (number) {
  return number * 2;
};
```

### Kilos to Pounds

```js
var kilosToPounds = function (kilos) {
  return kilos * 2.2;
};
```

### Area of a circle

```js
var circleArea = function (radius) {
  var pi = 3.1415;
  return pi * radius * radius;
};
```

## main

We can now define `main` as a function. However, we are not in charge of running it. Our starter code is running it for us whenever we click the submit button.

The `input` parameter of `main` is whatever gets typed in the input box.

The `return` value is what gets displayed in the grey box.

With this in mind, assign the value of `input` to myOutputValue to see this connection.

Then change the code again to assign the string value "wow hello" to `myOutputValue`.

## What is a function for?

We already defined a function as a collection of lines of code we can run.

Sometimes we just want to run a set number of lines of code collected together in our programs.

We are doing this in `main` - it is everything that we want to happen when the submit button is clicked.

Given the above function examples we can also say that a function can be a data calculation given some inputs. We can define an action as the kind of data it takes in and the kinds of outputs it gives.

## Errors

Your program will have errors. In fact, errors are an intrinsic part of developing software for everyone, not just begginers.

Let's talk about a few kinds of errors you might have seen so far.

Errors in your program will appear in the dev tools console.

Access the dev tools console by going to the top level menu and selecting view -&gt; developer -&gt; javascript console.

You should never code anything in the browser without having this pane open, otherwise you might not catch some errors.

You should always make sure either your page loaded with the tools already, or you reload the page after opening the dev tools or some errors might not get displayed.

There are 2 broad categories of errors. We can use the dev tools to fix both kinds.

### 1. Syntax errors / runtime errors

These are errors where you will see red text in the console.

Syntax errors can be divided into 2 kinds:

a. compilation errors - javascript looks at your code file for the first time and sees some syntax error that it can't move past.

Let's introduce a syntax error into your program.

Replace your `kilometersToMiles` function with the one below.

```js
var kilometersToMiles = function (distanceInKilometers) {
  var distanceInMiles = distanceInKilometers * 0.62;
  return distanceInMiles;
};
```

You'll get the error: `Uncaught SyntaxError: Unexpected token ')'`. This may look cryptic, but once you see this error a few times (and you definitely will) you'll understand what it is trying to tell you.

FIrst of all, the console is helpfully telling you which line of your code the error is occuring on.

Specifically in this case the error was a missing parenthese. Javascript knew because of the `function` keyword that a function was about to be defined. It was waiting for an opening left parenthese. When it found a right parenthese it knew for sure that something was wrong.

There are some spcific rules about why javascript can't know that a left parenthese was missing (rather than a right parenthese is incorrect) which are outside the scope of this course, but in practice, unfortunately, you as the programmer will have to know that "I didn't expect a right parenthese" infers that it was looking for a left parenthese.

A significant amount of the feedback you will get from the computer will be these semi-cryptic messages about why your code is wrong.

b. runtime errors - errors that happen when you run your porogram.

Replace this line in the starter code:

```js
var myOutputValue = 'hello world'();
```

You will see when you refresh the page that there is no red text.

Now click the submit button.

You'll get the error `Uncaught TypeError: "hello world" is not a function`. This is because there is a set of parentheses `()` instead of a semi-colon `;`.

Specifically javascript thinks the parentheses mean that you want to run some kind of function. This error isn't too common, but notice that for this type of error, javascript didn't let you know until it tried to run the code.

### 2. Logical / control errors

Some of the errors your program will have will not produce any red output on the console.

Replace your main function with this code:

```js
var main = function (input) {
  var myOutputValue = kilometersToMiles(input);
  myOutputValue;
};
```

This `main` function is missing the `return` keyword. Without it we can see the value output by the `main` function is the value `undefined`.

This code does not have any javascript _syntax_ errors - it is considered by the computer to be valid javascript code, but it does not **behave** the way we want.

### Debugging

Debugging is the process of diagnosing and fixing the errors you have in your program.

The debugging process can be relatively simple, in the case of type 1.a errors - syntax errors - javascript gives you red text in the console and you must start by going to the line that is mentioned in the console and then interpreting the true meaning of the error text given to you.

Debugging can become more complicated when you are trying to diagnose a logical error in your code- the computer is running the code without errors, but the behavior is not correct.

We'll talk about a few strategies you can use to diagnose and correct errors.

1. Control flow checking - is our code even runnning?

Functions give us the ability to execute code in a certain order, from certain places in our code that we specify. If something in our code seems incomplete or wrong, one of the first things we can check is to see if it is running at all.

#### console.log

`console.log()` is a function that we can run that will make text appear in the developer tools.

The thing we want to appear is passed an argument to `console.log` like so: `console.log("bananas!")`

Try writing some `console.log`s in all of the different structures of your code. Each `console.log` statement should be descriptive of where in the code you are. Also be careful not to make any errors here and describe something incorrectly!

We can use the fact that text is appearing in the dev tools console as an implicit signal that javascript it executing our code.

If our `console.log` statement appears in the dev tools console then we know that our code is reachable by opur program.

1. Value checking - is the value correct?

Another way we can use `console.log` is to pass it the variable values our program deals with.

Are the values in our program the ones we expect?

As below we can pass variable values to `console.log`:

```js
var main = function (input) {
  var myOutputValue = 'hello world';
  return myOutputValue;
};

var kilometersToMiles = function (distanceInKilometers) {
  var distanceInMiles = distanceInKilometers * 0.62;

  console.log('this is distance in miles value');
  console.log(distanceInMiles);

  return distanceInMiles;
};
```
