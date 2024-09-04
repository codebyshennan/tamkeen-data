# The `useEffect` Hook

## Learning Objectives
- Explain the purpose and function of the useEffect Hook in React, comparing it to lifecycle methods in class components.

- Implement useEffect to manage side effects in functional components, including API calls and subscription setups.

- Analyze the correct usage of useEffect dependencies and cleanup functions to optimize component performance and prevent memory leaks.

## Using the <a href="https://react.dev/reference/react/useEffect">`useEffect`</a> Hook

At Rocket we will use `useEffect` instead of the lifeCycleMethod `componentDidMount` for setting up subscriptions such as Firebase listeners and for data fetching. `useEffect` also allows us to perform functionality that was previously provided by `componentDidUpdate` and `componentWillUnmount`, but we will use that functionality less often.

1. As the docs mention, we can think of `useEffect` as running after the component renders.
2. A memory leak is when data that we use in parts of our apps is not properly cleaned up after the app stops using those parts. This can cause our apps to run slowly and even crash if the "leaks" cause our computers to run out of memory while running our apps. This will rarely happen to use in practice because JavaScript has automatic memory management and our apps will not be the most complex for now.
3. Returning a cleanup function from `useEffect` is optional and we will not use it often at Rocket.
4. No need to worry too much about "Optimizing Performance by Skipping Effects"; we will rarely need this and we can come back to it once we're more familiar with using `useEffect` and Hooks in general.

## Sample useEffect Hook

{% include youtube.html id="z4eeYjk57aM" %}
React Hooks useEffect

## <a href="https://react.dev/learn/lifecycle-of-reactive-effects" target="_blank">Component LifeCycles</a> and <a href="https://react.dev/learn/synchronizing-with-effects#how-to-write-an-effect" target="_blank">useEffect</a>

While developing React components it is possible to embed executable code within a component that runs on certain conditions. A common and useful pattern developers employ would be to call an API after the Component loads to get some information to display within the application. This can be controlled by the `useEffect` hook that is part of ReactJs.

1. Understand a components lifecycle and how they are "mounted", "updated" and "unmounted".
2. In React we use effects to describe how to synchronise  an external system to the current prop and state.
3. Understand how `useEffect` is implemented in components and what are the triggers that will force them to execute code
   - Import the `useEffect` hook from the react package, at the top of the component file
   - Call it at the top level of the component and put in your code, note state updates must be handled using conditional statements
   - Handle effect with dependencies, inside the dependency array, passed as the second argument to `useEffect`
   - Handle any effect clean ups by implemented a cleanup function within the effect.
4. <a href="https://react.dev/reference/react/useEffect" target="_blank">React's offical guide and additional reference</a> to the `useEffect` hook

<!-- ## Post-Class Exercises: Codecademy React 101

Complete all exercises in the following Codecademy lessons when they are assigned in the Rocket course schedule.

1. React Components
   1. <a href="https://www.codecademy.com/courses/react-101/lessons/your-first-react-component/exercises/hello-world-component" target="_blank">Your First React Component</a>
   2. <a href="https://www.codecademy.com/courses/react-101/lessons/react-components-advanced-jsx/exercises/render-multiline-jsx" target="_blank">Components and Advanced JSX</a>
   3. <a href="https://www.codecademy.com/courses/react-101/lessons/components-render-each-other/exercises/components-interacting-intro" target="_blank">Components Render Other Components</a>
   4. <a href="https://www.codecademy.com/courses/react-101/lessons/this-props/exercises/this-props-intro" target="_blank">this.props</a>
2. Hooks
   1. <a href="https://www.codecademy.com/courses/react-101/lessons/stateless-functional-components/exercises/stateless-functional-component-intro" target="_blank">Function Components</a>
   2. <a href="https://www.codecademy.com/courses/react-101/lessons/the-state-hook" target="_blank">The State Hook</a>
   3. <a href="https://www.codecademy.com/courses/react-101/lessons/the-effect-hook/exercises/function-component-effects" target="_blank">The Effect Hook</a> -->

## Additional Resources

The following resources are optional and can be used more for reference than upfront reading.

1. <a href="https://react.dev/learn/reusing-logic-with-custom-hooks" target="_blank">Re-using logic with Custom Hooks</a>
2. <a href="https://react.dev/reference/react" target="_blank">Hooks API Reference</a>