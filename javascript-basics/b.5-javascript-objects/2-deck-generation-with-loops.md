# Card Deck Generation with Loops

## Learning Objectives

By the end of this lesson, you should be able to:

* Create a representation of a standard deck of cards in JavaScript, where each card is an Object, and the deck is an Array of Objects, using a loop.

## Introduction

{% include youtube.html id="FG8QVeq5y-k" %}

At the end of the previous lesson, we shared a hard-coded card deck. This would work for our applications, but we may wish to use loops to generate this deck to be more concise and so we can more easily modify the deck if we need to. Creating the card deck with loops is also a useful exercise in identifying patterns and automating them with loops.

## Card Deck Generation Code

We define a `makeDeck` function that returns a generated standard 52-card deck. Read inline comments to understand the mechanics of this function.

```javascript
var makeDeck = function () {
  // Initialise an empty deck array
  var cardDeck = [];
  // Initialise an array of the 4 suits in our deck. We will loop over this array.
  var suits = ['hearts', 'diamonds', 'clubs', 'spades'];

  // Loop over the suits array
  var suitIndex = 0;
  while (suitIndex < suits.length) {
    // Store the current suit in a variable
    var currentSuit = suits[suitIndex];

    // Loop from 1 to 13 to create all cards for a given suit
    // Notice rankCounter starts at 1 and not 0, and ends at 13 and not 12.
    // This is an example of a loop without an array.
    var rankCounter = 1;
    while (rankCounter <= 13) {
      // By default, the card name is the same as rankCounter
      var cardName = rankCounter;

      // If rank is 1, 11, 12, or 13, set cardName to the ace or face card's name
      if (cardName == 1) {
        cardName = 'ace';
      } else if (cardName == 11) {
        cardName = 'jack';
      } else if (cardName == 12) {
        cardName = 'queen';
      } else if (cardName == 13) {
        cardName = 'king';
      }

      // Create a new card with the current name, suit, and rank
      var card = {
        name: cardName,
        suit: currentSuit,
        rank: rankCounter,
      };

      // Add the new card to the deck
      cardDeck.push(card);

      // Increment rankCounter to iterate over the next rank
      rankCounter += 1;
    }

    // Increment the suit index to iterate over the next suit
    suitIndex += 1;
  }

  // Return the completed card deck
  return cardDeck;
};
```

## Exercises

### Follow Along

Implement a card deck generation function from scratch. What might you do differently?
