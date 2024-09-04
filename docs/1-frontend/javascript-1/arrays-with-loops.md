# Loops with Arrays

## Learning Objectives

By the end of this lesson, you should be able to:

* Use the built-in `.length` array property.
* Use either a while or for loop to access or manipulate elements of an arbitrary-length array.

## Introduction

{% include youtube.html id="CMsjCb19_9s" %}

Often we want to perform actions on all elements of arrays of arbitrary length. We can use loops to run code on each array element sequentially.

The trick to using loops with arrays is to link the `counter` loop concept with the `index` array concept. Typically, the counter represents the current index of the array, and the length of the array is the maximum counter value plus 1. The "plus 1" is because array indexes are 0-indexed, thus the maximum valid array index is the array length minus 1.

## Looping Over Arrays

To get the length of an array, we can use the `.length` method. This method exists on every array.

```javascript
var letters = ['a', 'b', 'c'];
letters.length; // This returns 3, the number of elements in letters
```

In the following example we use `letters.length` to determine the number of loop iterations. We use `index` to access each element in the array sequentially, 1 element per loop iteration.

```javascript
// Index starts at 0, representing the 0th index of the array
var index = 0;
// We will iterate over the letters array
var letters = ['a', 'b', 'c'];
// Run the loop while index is less than the length of the array
while (index < letters.length) {
  // letters[index] represents a different element for each loop iteration
  var currentLoopLetter = letters[index];
  // Log the current element in each iteration
  console.log(currentLoopLetter);
  // Increment the index at the end of each iteration
  index = index + 1;
}
```

## Address Book Example

Let's create a practical example of loops with arrays using an address book. Our address book stores unique names in an array, i.e. will not add a name if it is already in the book.

```javascript
// Initialise an empty names array
var names = [];
var main = function (input) {
  // Set a boolean value found to a default value of false
  var found = false;
  
  // Loop over the names array, and set found to true if the input name already
  // exists in the names array
  var index = 0;
  while (index < names.length) {
    var currentName = names[index];
    if (currentName == input) {
      found = true;
    }
    index = index + 1;
  }

  // If no duplicate name was found, add the input name to the names array 
  if (!found) {
    names.push(input);
  }

  // Return the full array of names
  var myOutputValue = 'All names: ' + names;
  return myOutputValue;
};
```

## Exercises

### Follow Along

Implement the above code.
