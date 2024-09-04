# Errors and Debugging

## Learning Objectives

By the end of this lesson, you should be able to do the following:

- Define and differentiate between "syntax errors" and "logical errors".
- Know the difference between run-time and compile-time syntax errors.
- Understand strategies to debug errors.
- Utilise `console.log()` for effective debugging.

## Introduction

It has been told that on September 9th, 1947, [Grace Hopper](https://en.wikipedia.org/wiki/Grace_Hopper), an American computer scientist, found a moth in the Harvard Mark II computer’s logbook and reported the world’s first literal computer bug. However, the term “bug,” in the sense of technical error, dates back at least to 1878 and with Thomas Edison.

When programs throw errors that we didn’t expect to encounter, we call those errors bugs. Programmers call the process of updating the program so that it no longer produces bugs: debugging.

[Debugging a program]("https://preview.redd.it/2o2iq7pxia981.gif?format=mp4&s=0e7e3d534e257152b1929d3cd146b55d8232b02c")

Our programs will have errors. Errors are an intrinsic part of software development for everyone, not just beginners. What differentiates beginners from more-experienced engineers is the speed with which more-experienced engineers can triage and resolve errors.

Errors in our programs for Coding Fundamentals will appear in the [Chrome Dev Tools console](https://developer.chrome.com/docs/devtools/open/). Access the console by going to the Chrome menu bar and selecting View > Developer > JavaScript Console. Open the console before loading our webpages, and keep the console open when running code in the browser to catch errors as early as possible.

Unlike humans, computers follow strictly defined rules, cannot make sense of context, and do exactly as they are told. If we make grammatical mistakes when writing down instructions for them, or hope they read between the lines and understand what we hope to achieve, we are bound to run into errors.

There are 2 broad categories of errors: **syntax** and **logical** errors. Syntax errors are similar to grammatical errors in human languages, and logical errors are similar to giving poor directions to a lost tourist.

We can use the console to fix both kinds.

{% include youtube.html id="iWmlkW6uQ-0" %}

## Syntax errors

These are errors where we will see red text in the console. Syntax errors can be divided into 2 kinds: compile-time and run-time errors.

### Compile-Time Syntax Errors

Compile-time syntax errors are errors that can be detected before our code runs. These are similar to grammatical errors caught by auto-correct. VSCode will point out these errors in VSCode before we run our programs by highlighting the error code with red squiggly lines. If we try to run programs with compile-time errors, the programs will not run and we will see the errors in the Chrome console.

In some other programming languages, such as Java or C++, there is a compilation step before our code runs to translate human-readable code to machine code known as [binary](https://www.youtube.html/watch?v=wgbV6DLVezo).



Languages such as JavaScript and Python are [interpreted ](additional-resources-3.md#errors)programs where there isn't a dedicated compilation step, but the terminology compile-time can still be used to refer to time after the program is written but before it is executed.

Let's introduce a syntax error into our program. Replace our `kilometersToMiles` function with the one below.

```javascript
var kilometersToMiles = function distanceInKilometers) {
  var distanceInMiles = distanceInKilometers * 0.62;
  return distanceInMiles;
};
```

You'll get the error: `Uncaught SyntaxError: Unexpected token ')'`. This may look cryptic, but once you see this error a few times (and you likely will) you'll understand what it is trying to tell you.

First of all, the console is helpfully telling us which line of our code the error is occurring on.

Specifically in this case the error was a missing parenthesis. JavaScript knew because of the `function` keyword that a function was about to be defined. It was waiting for an opening left parenthesis. When it found a right parenthesis it knew that something was wrong.

There are specific rules why JavaScript might not detect that a left parenthesis (`(`) is missing instead of detecting that a right parenthesis (`)`) is incorrect. These rules are outside the scope of this course, but in practice, unfortunately, we as programmers will have to know that "I didn't expect a right parenthesis" implies we are missing a left parenthesis.

A significant amount of the feedback we will get from the computer will be semi-cryptic and we will try to understand this to debug errors.

![Humor error: Python doesn't use semicolons](https://preview.redd.it/5yhx7zsi7ug81.jpg?width=640&crop=smart&auto=webp&s=af94ec8e098d27f9b2ee3160c0e889674a3291e5)

### Run-Time Syntax Errors

Imagine sending a text message with no spelling or grammar errors, but a nonsensical one:

```
Singapore, lunch! you plane
```

Auto-correct won't catch any errors as each word is spelt correctly, grammatically valid and makes sense individually, but it's impossible to make sense of what is trying to be communicated.

Run-time syntax errors are syntax errors that are only detected _during_ when our programs are running. These types of errors are not possible in compiled languages because the entire program runs through a compiler before it can be executed.

JavaScript, being an interpreted language, code may only be read by the computer when it is executed. This produces 'run-time syntax errors' that are common in JavaScript. Fortunately, we have VSCode to help us detect these errors before we run our programs.

Let's create a run-time syntax error.

```javascript
var myOutputValue = 'hello world'();
```

1. Replace the relevant line in the starter code with the line above.
2. Refresh the starter code page in the browser. Observe that there is no red text in the console.
3. Click the Submit button. We should see the error `Uncaught TypeError: "hello world" is not a function` in the browser console.

This error happens because there is a set of parentheses (`()`) instead of a semi-colon (`;`) after our `'hello world'` string. JavaScript interprets parentheses to mean we want to run a function. While this error is uncommon, notice for this type of error that JavaScript did not alert us until it tried to run the code .

## Logical errors

Have you every received bad directions? Or instructions that are valid and sensical, but incomplete?

Logical errors are errors that arise from syntactically-valid code that does not _behave_ the way we want. These errors may not produce red console output because the program code does not have syntax errors.

### Debugging

Debugging is the process of diagnosing and fixing errors in our programs. The debugging process can be relatively straightforward. For syntax errors, our browser consoles output red error text. We can view the referenced line of code, interpret the error text, and fix our syntax. Debugging can become more complicated when diagnosing logical errors. Our code may run without errors, but its behaviour may be incorrect.

Here are 2 strategies to diagnose and correct errors.

1. Control-flow checking: Is our code running? Are the right blocks of code executing at the intended times?
2. Value checking: Do the values in our variables match what we expect?

### console.log

`console.log` is a a crucial debugging tool in JavaScript that we will use for both control-flow checking and value checking. It is a function we can run to make text appear in the browser console. We pass the text we want to appear as a parameter to `console.log` like this: `console.log('bananas!')`.

Try adding `console.log` statements to different parts of your code. Each `console.log` should indicate where in the code it is.

![This should be you.](https://i2.wp.com/programmingwithmosh.com/wp-content/uploads/2019/07/consolelogmeme.jpg?resize=442%2C442&ssl=1)

### Control-Flow Checking: Is Our Code Running?

Control-flow checking means checking if relevant parts of our code are being executed. Function syntax allows us to execute code in a specific order, or not at all. If our programs seem incomplete or wrong even though we have coded relevant logic into the program, it's possible we forgot to execute the relevant function, or our conditional logic (which we will learn in Module 6) failed to trigger the relevant code.

We can use `console.log` statements verify whether our browser is executing the relevant code. If our `console.log` statements appear in the browser console, we know our browser is executing that code.

Replace our starter code with the following and verify that we are running the `kmToMiles` function.

```javascript
var kilometersToMiles = function (distanceInKilometers) {
  console.log('running km to miles function');
  var distanceInMiles = distanceInKilometers * 0.62;
};

var main = function (input) {
  var myOutputValue = kilometersToMiles(input);
  return myOutputValue;
};
```

### Value Checking: Are the Values Correct?

Value checking means verifying that the intermediate variable values in our program match our expectations. Another way we can use `console.log` is to pass it the variable values we wish to verify.

Replace our starter code with the following. Are the values in our program the ones we expect?

```javascript
var kilometersToMiles = function (distanceInKilometers) {
  var distanceInMiles = distanceInKilometers * 0.62;
  return distanceInMiles;
};

var main = function (input) {
  var myOutputValue = kilometersToMiles(input);
  console.log('myOutputValue');
  console.log(myOutputValue);
  return myOutputValue;
};
```

### **`console.log` Formatting**

Formatting our console output can help make it more visible and decipherable to ourselves.

A few tips:

1. Be specific when describing each `console.log` statements, especially when we are logging variable values. Avoid generic labels like `number` or `output` and avoid repeating labels, which can obfuscate which variables are being logged.
2. Format output with special characters like `===`, `***`, or `###` to help output stand out.\
   `=== DISTANCE IN KM ===` stands out more than `distance in km`.

## Tips

1. **Mental models are the key to building your computational thinking skills.**
   1. Use each error as an opportunity to understand the system better. (But also know when to strategically move on - you won't achieve perfect knowledge).
2. **Expect lots of errors.**
   1. **You will run into bugs and errors.** This does not make you a poor or lousy programmer.
   2. Errors are the practice and process of programming. Use them to learn more. Be playful with the system and don't be afraid to break things.
3. **Formulate questions.** Practice asking questions when you are working on a problem:
   1. What have you tried?
   2. What is the context of the error? When do you see it, what is the incoming data and how does it affect the output?
   3. What do you think is happening based on the facts?
4. **Talk to yourself. This is called**[ **rubber duck programming**](https://rubberduckdebugging.com)**.**

<img src="https://cdn.shopify.com/s/files/1/0243/9469/4761/articles/rubber-duck-debugging_670x.jpg?v=1558873639" alt="quack quack" data-size="original">

## Exercise

1. You'll have plenty.
