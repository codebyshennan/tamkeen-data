# DOM Manipulation

## Learning Objectives:

By the end of this lesson, you should be able to:

- Create, select and manipulate HTML elements using DOM-related methods.
- Write JavaScript logic that manipulates the DOM based on user input.

## Introduction

We have seen how we can manipulate the DOM via JavaScript in the console. Now let's take a look at how we can pre-program DOM manipulation to happen based on user input.

Conceptually, we are still just defining **inputs** and **outputs.** Now inputs include how the user interacts with the browser, which can include mouseovers, scrolls, drag and drops, clicks or double clicks, and output is manipulation of the Document that the user is interacting with.

## Inputs and Outputs

#### Defining an Input

{% include youtube.html id="6fEp-k32Tv8" %}

An input to our JavaScript program is called an **event**. Similar to how we can define functions as blocks of code that will only run when the function is called, we can write code that will only run when the user interacts with the browser via an **event**. We do this with the provided `.addEventListener()` method. The method takes in 2 inputs:

1. The [event type](https://developer.mozilla.org/en-US/docs/Web/Events)
2. The code block to be executed when the event is triggered, **defined as a function**.

Some useful events include:

- click
- dblclick
- keypress
- keydown
- keyup

You can find a more complete list with explanations of each event on [this page](https://www.w3schools.com/jsref/dom_obj_event.asp).

Let's go back to basics, and print out "Hello, world!" in the console whenever a button is clicked. The syntax for that might look something like this:

```javascript
var newButton = document.createElement('button');
newButton.innerText = 'Hello';
newButton.addEventListener('click', function () {
  console.log('hello, world!');
});
document.body.appendChild(newButton);
```

This means we can program the browser in such a way that when the user makes an action, the browser calls our function - and this is exactly how our `main` function works. Except that we aren't just limited to clicking on the submit button or getting the value the user typed into an input field/textbox.

The function for an event is referred to as a **callback** function. In the case of an event listener, the callback function takes the triggered event as an input parameter and does whatever actions it was declared to do due to this trigger. Now we can look at how this was defined for you in index.html of the starter code:

```javascript
      // Declare and define a variable that represents the Submit Button

      var button = document.querySelector("#submit-button");

      // When the submit button is clicked, access the text entered
      // by the user in the input field

      // And pass it as an input parameter to the main function as
      // defined in script.js

      button.addEventListener("click", function () {

        // Set result to input value
        var input = document.querySelector("#input-field");

        // Store the output of main() in a new variable
        var result = main(input.value);

        // Display result in output element
        var output = document.querySelector("#output-div");

        output.innerHTML = result;

        // Reset input value
        input.value = "";
      });
```

## Advanced Applications in the Browser

{% include youtube.html id="mT-oeRViW9M" %}

The pattern of:

1. Being able to create any representation of data on the screen
2. Have complete control over what actions the user can take at any time

means that your app should have complete control over every element on the screen.

Modern development for applications that run in the browser ( the browser is the front-end ) are libraries and frameworks that allow programmers to manage all of these elements and inputs.

Some popular libraries include [ReactJS](https://reactjs.org), [Angular](https://angular.io), and [Vue](https://vuejs.org).

## More Examples

Read through the comments to see what's actually happening.

```html
<p>
    <input id="starter-ex"/>
</p>
<button id="starter-button">submit me</button>
```

```javascript
// make a variable out of the input and button
var input = document.querySelector('#starter-ex');
var button = document.querySelector('#starter-button');

// call this function
var myButtonClicked = function () {
  // get the current value that's been typed into the input
  var typedValue = input.value;

  // create a new h2
  var newHtwo = document.createElement('h2');

  // set the text inside this new element
  newHtwo.innerText = typedValue;

  // make the h2 appear on screen
  document.body.appendChild(newHtwo);

  // empty out the HTML input
  input.value = '';
};

// say which function to call *when* the user clicks the button
button.addEventListener('click', myButtonClicked);
```

## Exercises

### Follow Along

1. Implement the above code.
2. Write logic that grabs and references HTML elements, and change the background color of the `container` element every time the submit button is clicked.
