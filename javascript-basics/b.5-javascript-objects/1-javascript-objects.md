# JavaScript Objects

## Learning Objectives

By the end of this lesson, you should be able to:

* Explain what an Object is in JavaScript.
* Explain when and why to use an Object datatype.
* Explain the differences between Objects vs Arrays.
* Declare an Object using JavaScript syntax.
* Access, add, or replace attributes in an Object.

## Introduction

{% include youtube.html id="11z4g8BOxXg" %}

In this module we will introduce a new data structure called the JavaScript **Object**. This data structure is called different names in different contexts. For example, JS Objects are known as "hash tables" in computer science, "dictionaries" in Python, and "HashMaps" in Java. The JS Object is **not** the same "object" concept referred to in Object Oriented Programming.

We will use Objects to implement card games. It turns out that cards are easily represented by Objects and a clear demonstration of why Objects can be useful.

### How to Represent a Card Deck?

{% include youtube.html id="vSIm-Y9KANg" %}

What would be the best way to represent a deck of cards in code? A deck of cards is a set of data (cards) that is of the same type (each card has the same attributes). This implies an array. We could create an array like the following.

```javascript
var cardDeck = ['nine of hearts', 'ten of hearts', 'jack of hearts'];
```

Is this a good representation of a card deck? Let's think abstractly about what attributes belong to a card deck.

#### Deck Order

A deck of cards is implicitly ordered. An array was a good choice to represent this.

#### Card Attributes

1. Each card has a rank. Our array of strings stores this rank in the sub-strings "nine", "ten", or "jack". Card ranks have order that we do not represent in our strings, e.g. `queen > jack`.
2. Each card has a suit. Our array of strings stores this suit in the sub-string "hearts".

How would we obtain rank and suit attributes of cards in our deck? In our above example, we could perform string manipulation to extract rank and suit values, but we would subsequently also have to write logic to compare rank strings to determine card order. Let's see if we can represent card attributes in a more convenient manner.

### Objects Store Compound Data

Objects are perfect for representing single units of data (e.g. cards) that themselves contains smaller units of data that are attributes of the larger unit. The attributes of an Object need not be the same data type- they can be any data type, and each attribute can have its own data type. The following example is an Object representation of a single playing card.

```javascript
// Objects are defined with curly-brace syntax and can be stored in variables.
var playingCard = {
  // Each attribute is defined with a key and a value.
  // The key is before the colon (:), and the value is after the colon.
  // Attributes are separated by a comma and newline.
  // Use a number to represent rank for easy comparison with other cards
  rank: 11,
  suit: 'heart',
  // Store an additional "name" attribute to correctly name face cards and aces.
  name: 'jack',
};
```

The example encodes 3 attributes of a playing card. `rank` is a number to facilitate comparison with other cards. `suit` is still a string but now easily accessible without string manipulation. `name` is used for display purposes, to quickly convert the `rank` number into a string. This is especially helpful for face cards whose names are not the English word of the `rank` number.

### How to Access Object Attributes

We can use the `.` operator to access Object attributes. In the following code snippet, we access each of the `playingCard` Object's attributes with the `.` operator. We can also access Object attributes using `[]`notation, but this will typically not be necessary.

```javascript
// Access the rank attribute
playingCard.rank;
// Access the suit attribute
playingCard.suit;
// Access the name attribute
playingCard.name;
```

### Objects with Arrays

We can combine objects and arrays to store more complex data types, such as a deck of cards. Each card is represented by an object, and the set of cards is represented by an array.

```javascript
var cardDeck = [
  {
    rank: 1,
    suit: 'heart',
    name: 'one',
  },
  {
    rank: 2,
    suit: 'heart',
    name: 'two',
  },
  // ...
];
```

### Objects, Arrays, and Loops

Given the above card deck representation, we can use a loop to access all cards in a deck.

```javascript
// Initialise index to 0 to start from the beginning of the array
var index = 0;
// Define loop condition to loop until index is the length of cardDeck
while (index < cardDeck.length) {
  // Access attributes of each card with dot notation.
  console.log(cardDeck[index].name);
  // Construct a string using attributes of each card object
  var cardTitle = cardDeck[index].name + ' of ' + cardDeck[index].suit;
  // Log the string
  console.log(cardTitle);
  // Increment the card index
  index = index + 1;
}
```

Inside the loop we can access a specific card's attributes using dot notation on that card's object in the array.

```javascript
// Get the name of the 1st card
cardDeck[0].name;
```

### Objects vs. Arrays

When should we use Objects versus Arrays? Here is a summary.

|                                                 | Objects                                                                                      | Array                                                                                                                                                                |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Is the data ordered or unordered?               | Unordered                                                                                    | Ordered                                                                                                                                                              |
| What kind of data is stored inside?             | Attributes that represent different types of data, e.g. colours, suits, and ranks for cards. | Elements that are all the same type of data, both in terms of data type (e.g. boolean, number, string, Object, Array) and data meaning (e.g. all elements are cards) |
| What operations do you want to do on this data? | Access a single piece of the data: `card.rank` `card.suit`                                   | Do the same operation to every piece of data you have (in a loop).                                                                                                   |

### Card Shuffling

{% include youtube.html id="8DRwHSiyGe8" %}

We can shuffle elements in an array by swapping random elements in the array repeatedly. We can apply this concept to shuffling cards, where each element in a card deck array is an object representing a playing card. The following code simulates card shuffling and will be useful for the card games we will implement.

```javascript
// Get a random index ranging from 0 (inclusive) to max (exclusive).
var getRandomIndex = function (max) {
  return Math.floor(Math.random() * max);
};

// Shuffle the elements in the cardDeck array
var shuffleCards = function (cardDeck) {
  // Loop over the card deck array once
  var currentIndex = 0;
  while (currentIndex < cardDeck.length) {
    // Select a random index in the deck
    var randomIndex = getRandomIndex(cardDeck.length);
    // Select the card that corresponds to randomIndex
    var randomCard = cardDeck[randomIndex];
    // Select the card that corresponds to currentIndex
    var currentCard = cardDeck[currentIndex];
    // Swap positions of randomCard and currentCard in the deck
    cardDeck[currentIndex] = randomCard;
    cardDeck[randomIndex] = currentCard;
    // Increment currentIndex
    currentIndex = currentIndex + 1;
  }
  // Return the shuffled deck
  return cardDeck;
};
```

### High Card

{% include youtube.html id="ZMDQwazN8I0" %}

High Card is a card game where each player draws a random card, and the player with the highest card wins. Below is a sample implementation of High Card. We will soon be creating other card games, combining all we have learned so far in SWE Fundamentals.

In the following implementation, we use the JS array `pop` method to draw a random card from the deck. `pop` removes and returns the last array element (i.e. draws a card from the top of the deck). We can rely on `pop` to draw a random card because the deck is already shuffled.

After drawing cards, we can use the cards' `rank` attributes to compare which is higher.

```javascript
// Initialise the card deck representation as an array of objects
var deck = [
  // card1,
  // card2,
  // ...
];

// Shuffle the deck and save it in a new variable shuffledDeck
// to communicate that we have shuffled the deck.
var shuffledDeck = shuffleCards(deck);

var main = function (input) {
  // Draw 2 cards from the top of the deck
  var computerCard = shuffledDeck.pop();
  var playerCard = shuffledDeck.pop();

  // Construct an output string to communicate which cards were drawn
  var myOutputValue =
    'Computer had ' +
    computerCard.name +
    ' of ' +
    computerCard.suit +
    '. Player had ' +
    playerCard.name +
    ' of ' +
    playerCard.suit +
    '. ';

  // Compare computer and player cards by rank attribute
  // If computer card rank is greater than player card rank, computer wins
  if (computerCard.rank > playerCard.rank) {
    // Add conditional-dependent text to the output string
    myOutputValue = myOutputValue + 'Computer wins.';
    // Else if computer card rank is less than player card rank, player wins
  } else if (computerCard.rank < playerCard.rank) {
    myOutputValue = myOutputValue + 'Player wins!';
    // Otherwise (i.e. ranks are equal), it's a tie
  } else {
    myOutputValue = myOutputValue + "It's a tie.";
  }

  // Return the fully-constructed output string
  return myOutputValue;
};
```

## Hard-Coded Card Deck

1 acceptable way to initialise a standard 52-card deck in our code is to hard-code it, i.e. manually specify each card and each of its attributes in code. Use this hard-coded deck for this module's exercises below.

```javascript
var deck = [
  {
    name: 'ace',
    suit: 'hearts',
    rank: 1,
  },
  {
    name: '2',
    suit: 'hearts',
    rank: 2,
  },
  {
    name: '3',
    suit: 'hearts',
    rank: 3,
  },
  {
    name: '4',
    suit: 'hearts',
    rank: 4,
  },
  {
    name: '5',
    suit: 'hearts',
    rank: 5,
  },
  {
    name: '6',
    suit: 'hearts',
    rank: 6,
  },
  {
    name: '7',
    suit: 'hearts',
    rank: 7,
  },
  {
    name: '8',
    suit: 'hearts',
    rank: 8,
  },
  {
    name: '9',
    suit: 'hearts',
    rank: 9,
  },
  {
    name: '10',
    suit: 'hearts',
    rank: 10,
  },
  {
    name: 'jack',
    suit: 'hearts',
    rank: 11,
  },
  {
    name: 'queen',
    suit: 'hearts',
    rank: 12,
  },
  {
    name: 'king',
    suit: 'hearts',
    rank: 13,
  },
  {
    name: 'ace',
    suit: 'diamonds',
    rank: 1,
  },
  {
    name: '2',
    suit: 'diamonds',
    rank: 2,
  },
  {
    name: '3',
    suit: 'diamonds',
    rank: 3,
  },
  {
    name: '4',
    suit: 'diamonds',
    rank: 4,
  },
  {
    name: '5',
    suit: 'diamonds',
    rank: 5,
  },
  {
    name: '6',
    suit: 'diamonds',
    rank: 6,
  },
  {
    name: '7',
    suit: 'diamonds',
    rank: 7,
  },
  {
    name: '8',
    suit: 'diamonds',
    rank: 8,
  },
  {
    name: '9',
    suit: 'diamonds',
    rank: 9,
  },
  {
    name: '10',
    suit: 'diamonds',
    rank: 10,
  },
  {
    name: 'jack',
    suit: 'diamonds',
    rank: 11,
  },
  {
    name: 'queen',
    suit: 'diamonds',
    rank: 12,
  },
  {
    name: 'king',
    suit: 'diamonds',
    rank: 13,
  },
  {
    name: 'ace',
    suit: 'clubs',
    rank: 1,
  },
  {
    name: '2',
    suit: 'clubs',
    rank: 2,
  },
  {
    name: '3',
    suit: 'clubs',
    rank: 3,
  },
  {
    name: '4',
    suit: 'clubs',
    rank: 4,
  },
  {
    name: '5',
    suit: 'clubs',
    rank: 5,
  },
  {
    name: '6',
    suit: 'clubs',
    rank: 6,
  },
  {
    name: '7',
    suit: 'clubs',
    rank: 7,
  },
  {
    name: '8',
    suit: 'clubs',
    rank: 8,
  },
  {
    name: '9',
    suit: 'clubs',
    rank: 9,
  },
  {
    name: '10',
    suit: 'clubs',
    rank: 10,
  },
  {
    name: 'jack',
    suit: 'clubs',
    rank: 11,
  },
  {
    name: 'queen',
    suit: 'clubs',
    rank: 12,
  },
  {
    name: 'king',
    suit: 'clubs',
    rank: 13,
  },
  {
    name: 'ace',
    suit: 'spades',
    rank: 1,
  },
  {
    name: '2',
    suit: 'spades',
    rank: 2,
  },
  {
    name: '3',
    suit: 'spades',
    rank: 3,
  },
  {
    name: '4',
    suit: 'spades',
    rank: 4,
  },
  {
    name: '5',
    suit: 'spades',
    rank: 5,
  },
  {
    name: '6',
    suit: 'spades',
    rank: 6,
  },
  {
    name: '7',
    suit: 'spades',
    rank: 7,
  },
  {
    name: '8',
    suit: 'spades',
    rank: 8,
  },
  {
    name: '9',
    suit: 'spades',
    rank: 9,
  },
  {
    name: '10',
    suit: 'spades',
    rank: 10,
  },
  {
    name: 'jack',
    suit: 'spades',
    rank: 11,
  },
  {
    name: 'queen',
    suit: 'spades',
    rank: 12,
  },
  {
    name: 'king',
    suit: 'spades',
    rank: 13,
  },
];
```

## Exercises

### Follow Along

Implement the High Card code above to get a working High Card game.

### Highest of 2 Cards

Change the High Card program so that the player and computer each draw 2 cards instead of 1. The player with the highest of any of the cards wins.

## Reference

### Object Dereferencing: `myObj.name` vs `myObj[name]` vs `myObj['name']`

`myObj.name` and `myObj['name']` are equivalent.

`myObj[name]` returns the value that corresponds to the key with the value stored inside the variable `name`. For example, if we declare `var name = 'Luke';`, then `myObj[name]` is equal to `myObj.Luke`.

We cannot use `.` notation for numbers. For example, for an object key with value `'1'`, we cannot do `myObj.1`. We must do `myObj[1]` or `myObj['1']`, where JS auto-converts `1` in the former example to `'1'` since object keys can only be strings.
