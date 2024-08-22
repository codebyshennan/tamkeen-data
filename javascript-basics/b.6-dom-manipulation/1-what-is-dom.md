# The Document Object Model

## Learning Objectives

By the end of this lesson, you should be able to:

- Briefly explain the Document Object Model (DOM).
- Reference HTML elements using classes and IDs
- Perform basic DOM manipulation in the browser console.

## Overview

Now we'll describe more about the browser environment we've been working with. This section is meant as a crash course on the rest of the browser system.

Here we can distinguish between **domain knowledge** - technical knowledge of a specific system, and **general programming knowledge**, which we've been focusing on during this course.

## Introduction

In next lesson, we will learn how to make our web applications more interactive by using JavaScript logic to change the previously-static elements that are rendered by the browser. We can effectively write programs that change what the user sees. We can tell the browser "change the background color of this element when the submit button is clicked" or "disable this button if the user runs out of money" or, if you are feeling nefarious, "make this pop-up appear".

In order to do that, we must first be able to _reference_ HTML elements that have already been rendered and link them to JavaScript variables, so that we can manipulate them the same way we manipulated data previously.


Similar to how we started with entering commands into the console, we will first explore and experiment with the DOM in the browser console. 

{% include youtube.html id="AP8YcJ9MR64" %}

## DOM - Document Object Model

By now you should be familiar that while the browser primarily reads `index.html` , the entire web page that is displayed is built from multiple files together, such as `script.js` and `styles.css`. Even within `index.html`, there is the `<head>` and `<body>` elements, each of which contain child elements. Data from various sources together define the information and structure of the document that is processed and displayed by the browser.

![](https://www.freecodecamp.org/news/content/images/2021/01/images.png)

The Document Object Model, or DOM, is a **representation** of the contents of a webpage. Specifically, it's a tree-like representation of nodes and objects. By representing the entire document in an object-oriented way, we can use JavaScript to interact with and manipulate the displayed page in a dynamic way.

When we want our JavaScript to take in any inputs from the user or display any outputs on the browser, we have to use some part of the DOM functionality.

When the document is processed by the browser, a set of objects are created that represents the document, along with pre-defined methods and functions that we can use.

We've already seen one: `console.log()`. To be specific, `console` is an _object_ of the DOM created by the browser, which has a pre-defined `.log()` _method._ Other `console` methods include \*\* `.clear()` \*\* or `.table()`_._

Besides `console,` the other major objects that represent the DOM are `window` and `document`. We will next explore the `document` object and the `document.querySelector()` method.


Open the console and type in the variable names `window` and `document` to see the values in the console. Click the triangles to the left of the output to see inside them. What is there?


## Accessing Elements in the DOM

{% include youtube.html id="eLYFvfY1lKU" %}

Not every node in the DOM "tree" is pre-defined as an object. You can use JavaScript object notation to access nested _nodes_ in the document object: try `document.body` or `document.head`, or more generally: `document.children`. We can then use nested object notation to access child elements, so `document.body.children[0]` will access the first nested element in the `<body>`. This clearly isn't intuitive, what if we were looking for a specific element?

`document.querySelector()` allows us to do just that. It takes in, as input, a search query and returns the first element in the DOM that matches. If we wanted to access a paragraph element, for example, we could use `document.querySelector("p");`

If we wanted to access a specific paragraph, we can do so by referencing its id or class by using a . or # prefix, respectively.

`document.querySelector("#submit-button")` will return the element that has the id "submit-button".

`document.querySelector(".card")` will return the first element that has the "card" class.

We can store the returned element in it's own variable, which will then allow us to access element properties or use other DOM methods on the element itself. Putting it all together:

```javascript
var button = document.querySelector('#submit-button');
button.innerText;
button.innerText = 'Click Me!';
```

Can you guess what `.innerText` is or does? What happens when you execute the code above in the console?

### Adding Elements to the DOM

{% include youtube.html id="9E2wbG8Y6OY" %}

We have already seen how to use `document.querySelector()` to access elements that are already on the page. What if we wanted to create a new element and add it to the document, without having to edit `index.html`?

#### Creating an Output

We can use the DOM to create elements and put them on the page.

```javascript
var coolParagraph = document.createElement('p');
coolParagraph.innerText = 'hey, cool!!!1!';
```

After we've created this element we need to tell the DOM to put it into the page.

```javascript
document.body.appendChild(coolParagraph);
```

## Exercise

1. Change the background colour of the `<div>` with class `"container"` using the DOM.
   1. Hint: you may want to access the `style` attribute of the `<div>` element.
2. Create and add a new HTML element to the page using the console.
