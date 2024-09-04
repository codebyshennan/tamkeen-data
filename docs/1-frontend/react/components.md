# React Components and Hooks

## Learning Objectives

- Construct React components following the core rules of component creation and rendering.

- Implement props to pass data from parent to child components in a React application.

## <a href="https://react.dev/learn/your-first-component" target="_blank">React Components</a>

The concept of Components is core to React, they are the foundation of our React code, we use components to build user interfaces. This makes them ideal starting points once you've understood the basics of React JSX. React allows developers to combine markup, CSS and JavaScript into components that could be reusable UI elements within your application. Similarly to HTML tags we can compose, order and nest components to develop full pages within React applications. When building a Component follow these rules:

1. Export the Component
2. Define the Component function
3. Add any markup required, this is what we want to display within the Component
4. Render the Component onto the React Application by nesting it into the App.jsx
5. User-defined components must be capitalised. Otherwise React will think they are HTML tags.

## <a href="https://react.dev/learn/passing-props-to-a-component" target="_blank">Component Props</a>

React components can communicate with each other via props, a parent may pass information down to child component and this data always flows downwards. There will be some familiar props associated to JSX tags, but you you can actually pass anything from a parent its children.

1. To pass props specify the information that should be passed from the parent to the child.
2. Access the prop information that was passed to the child component and set default values if required.
3. You can pass anything as a prop from a parent component to a child, you can access props by destructuring the individually passed sets of information or by referring to props.
4. Props are immutable, checkout <a href="https://react.dev/learn/passing-props-to-a-component#how-props-change-over-time" target="_blank">this example</a> to see how they can change over time.

## Introduction to React Hooks

> **Note**: *The following introduction was written around 2022 when the React community was in the midst of transitioning to using hooks and functional components. In 2024, we preserve the original text for historical context.*

> React Hooks are a newer and more efficient syntax for React Components. They allow us to write all components as functional components, and use so-called "Hook" functions to replace class component functionality such as state management and lifecycle methods. Components with hooks are functionally the same as class components.
> 
> Rocket recommends using Hooks in our exercises and projects from now on because Hooks are a cleaner syntax and the <a href="https://reactjs.org/docs/hooks-faq.html#should-i-use-hooks-classes-or-a-mix-of-both" target="_blank">React team recommends using Hooks for new projects</a>. The React team is <a href="https://beta.reactjs.org/" target="_blank">re-writing the official React tutorials to use Hooks</a>. React Router v6 (the latest and greatest version of React Router) that we are about to learn only supports React Hooks syntax natively, and Rocket recommends using Hooks to enable us to use the latest React Router features.
>
> Many companies are still using <a href="https://legacy.reactjs.org/docs/react-component.html" target="_blank">Class based React Components</a> in their code. It will be important for us to still understand class Components, but know how to write Components with Hooks for new code to take advantage of latest React functionality. If you want to explore how to use state and lifecycles in React Class based Components you can read <a href="https://legacy.reactjs.org/docs/state-and-lifecycle.html" target="_blank">this documentation</a>.

## <a href="https://react.dev/reference/react">Hooks</a> at a Glance


1. React Hooks provide all the functionality we need from React Class based Components that we didn't previously have with Functional Components of the past.
2. `useState` hook replaces `this.state` and `this.setState` with a new pair of variables for getting and setting a specific state value.
3. Note how the React team encourages us to use multiple `useState` Hooks in the same component for each type of state. There is no need to store all of a component's state in a single `this.state` object like we did previously.
4. Returning a cleanup function from `useEffect` is advanced functionality and we will not be using it as often at Rocket
5. We will not be writing custom Hooks at Rocket; feel free to skim through the "Building Your Own Hooks" section
6. We will learn about `useContext` and `useReducer` hooks in a later submodule.

## <a href="https://react.dev/reference/rules/rules-of-hooks" target="_blank">Rules of Hooks</a>

There are some general rules that must be applied when using hooks in React. Take a look at the documentation for details, but here is the summary:

1. <a href="https://react.dev/reference/rules/rules-of-hooks#only-call-hooks-at-the-top-level" target="_blank">Only call Hooks at the top level</a> – Don’t call Hooks inside loops, conditions, or nested functions. Instead, always use Hooks at the top level of your React function, before any early returns.
2. <a href="https://react.dev/reference/rules/rules-of-hooks#only-call-hooks-from-react-functions" target="_blank">Only call Hooks from React functions</a> – Don’t call Hooks from regular JavaScript functions.

