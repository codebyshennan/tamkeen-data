# Handling Events

## Learning Objectives

- Identify the steps required to handle events in React applications using JSX.

- Implement event listeners and callback functions to respond to user interactions in React components.

- Apply techniques to manage event propagation and prevent default behaviors in React event handling.

## <a href="https://react.dev/learn/responding-to-events" target="_blank">Handling Events</a>

When users click on a button they expect some result, or some change, take the counter example if you click on the button the state increments. To develop these types of functionality within our React applications we will need to handle events, but assigning an event listener to our elements and define some callback functions to handle the event in a meaningful way. In our JSX code we need to follow these steps to handle events.

1. How do you handle events?
   1. Attach the event listener to the element, such as an `onClick` or `onBlur` event.
   2. Define the callback function that will handle said event, accessing the event data if required. Remember to pass this function to the event handler.
   3. Update state within this callback function by calling an updater function, run a side effect or execute any code.
2. "Events on DOM elements" are the same as <a href="https://www.w3schools.com/js/js_events.asp" target="_blank">JavaScript events or HTML events</a>. JS events allow us to perform logic on events that happen on our web pages such as mouse clicks. React supports a <a href="https://react.dev/reference/react-dom/components/common#common-props" target="_blank">wide range of events.</a>
3. Remember that functions that are passed to event handlers must be passed and not called, you can wrap the given function in an anonymous function if you need to pass in arguments.
4. You are able to stop event propagation as well as the default action.
