# Using `<form>`, `<input>`, `<select>` and `<textarea>`

## Learning Objectives

- Construct functional forms in React using input, select, and textarea components.

- Implement controlled form components by managing their state with React hooks and handling associated events.

- Develop a form submission handler that captures and processes user input data without page refresh.

## <a href="https://react.dev/reference/react-dom/components#form-components" target="_blank">Forms</a>, <a href="https://react.dev/reference/react-dom/components/input" target="_blank">input</a>, <a href="https://react.dev/reference/react-dom/components/select" target="_blank">select</a>, <a href="https://react.dev/reference/react-dom/components/textarea" target="_blank">textarea</a>

HTML forms and their elements afford developers the opportunity to capture user information as they interact with our application. Forms can be composed of a combination of inputs, selects and textareas, when wrapped in a form tag the values of these inputs can be extracted as form data. Or they can be maintained purely using state. Employ React element such as input, select and text areas to give users different input capabilities.

1. An <a href="https://www.w3schools.com/html/html_forms.asp" target="_blank">HTML form</a> is an HTML element that either refreshes the page or navigates to a new page when the user submits the form. We often do not want this behaviour in React, opting to update UI on submit without refresh. To disable "refresh on submit" behaviour, Rocket recommends the technique in the React docs, to provide the `form` a `handleSubmit` callback function that calls `event.preventDefault`.
2. When handling the submit method from a form you should target the `event.target.value` to get the current value of an input.
3. You must control the form elements that you use within a React component, do this by passing the value prop to it and handling the associated event. <a href="https://react.dev/reference/react-dom/components/input#controlling-an-input-with-a-state-variable" target="_blank">Have an in-depth look here</a>.
4. Note how you will need to create state to capture form data.

<!-- ## Post-Class Exercises: Codecademy React 101

Complete all exercises in the following Codecademy lessons when they are assigned in the Rocket course schedule.

1. <a href="https://www.codecademy.com/courses/react-101/lessons/react-forms/exercises/react-forms-intro" target="_blank">React Forms</a> -->


