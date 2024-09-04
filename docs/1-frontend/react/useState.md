# The `useState` Hook

## Learning Objectives

- Apply the useState hook to create and manage state variables in React functional components.

- Apply the recommended naming convention for useState variables to improve code readability.

- Implement effective state management practices, including grouping related state variables and updating state based on its current value.

## Using the <a href="https://react.dev/reference/react/useState">`useState`</a> Hook

We will use the `useState` Hook most often. Clear explanations of what the `useState` Hook is and how to use it.

> Rocket strongly recommends following the naming convention of `X` and `setX` as the de-structured state variable names from `useState`. This makes our code more readable because other engineers will immediately know what each variable is for.

### Sample `useState` Hook

{% include youtube.html id="eb6HKsxHoas" %}
React Hooks `useState`

### <a href="https://react.dev/learn/managing-state" target="_blank">React State management, `useState`</a>

While creating your application and enlarging functionalities you will find that you are developing your components and states into much larger and complex files than you had before. IT is important to manage state effectively and reduce redundant or duplicated states within your application.

1. When using `useState` to manage component state Rocket advises following these steps
   - Import the `useState` hook from the react package, at the top of the component file
   - Within your functional component declare a state variable and the updater function using the `useState` hook. As shown on <a href="https://react.dev/learn/managing-state#reacting-to-input-with-state" target="_blank">line 4</a>.
   - The initialValue of the state is passed into `useState`.
   - You can update this variable using the updater function.
2. We will take a longer look onto `useContext` and `useReducer` later in this course.
3. When developing your state, group information if two state variables always change together, therefore unify them into a single state variable. Here are some more rules regarding <a href="https://react.dev/learn/choosing-the-state-structure" target="_blank">state structure</a>.
4. When updating a state variable its possible to use its current value, this is done by invoking a callback function on the updater function. Take a look at <a href="https://react.dev/reference/react/useState#updating-state-based-on-the-previous-state" target="_blank">these examples here</a>.
