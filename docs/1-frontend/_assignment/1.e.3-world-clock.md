# Assignment: World Clock

## Learning Objectives

1. Install and configure packages in an NPM project and initiate an NPM application.
2. Utilize JavaScript and HTML syntax within JSX to develop dynamic user interfaces.
3. Analyze the process by which React interacts with the DOM to render components on a webpage.
4. Implement useState hooks in React components to manage and manipulate the user interface.
5. Utilize component lifecycle methods to execute logic during the mounting and unmounting phases of a React component.
6. Encapsulate UI elements within React components and pass data to them using props.

## Introduction

We will build a collection of clocks with different time zones to demonstrate the fundamentals of React and Node apps.

## Starter Code

### Clone starter code

Fork and clone <a href="https://github.com/rocketacademy/world-clock-3.2" target="_blank">Rocket's World Clock repo</a> (Rocket-themed React Vite application). Run `npm install` to install default packages our app needs to run, and run `npm run dev` to start the app next open your browser and navigate to <a href="http://localhost:5173/" target="_blank">http://localhost:5173</a>.

### Understand starter code

1. `README.md` contains instructions for running the app
2. `package.json` lists the packages our app needs to run as well as their versions in `dependencies`, its also contains `scripts` that ViteJs provides that are used to run and maintain the application. We can add as many new dependencies and scripts as we want, but we will not add any for this exercise to keep our app simple. We will not strictly need to, but if you are curious, read <a href="https://docs.npmjs.com/cli/v8/configuring-npm/package-json" target="_blank">`package.json` docs</a> to understand its other attributes in more detail.
3. `package-lock.json` lists all dependencies of the packages that are listed in `package.json` as well as their versions and other required dependenies. This makes sure all dependency versions are standardised for consistency in running our app. We should never alter this file manually.
4. `node_modules` contains all dependency files installed by `npm install`. We should never modify `node_modules` directly nor commit it to GitHub because its contents can be re-generated on-demand with `npm install`, provided you've shared the `package.json`.
5. `src` contains our source code, i.e. our app logic. `src/main.jsx` renders our React app's root component into the root HTML page, and `App.jsx` defines our React app logic.
6. `public` contains static files to load the app website, including its favicon (icon in tab bar) root HTML page and manifest for SEO purposes. Remember, we place our image assets here.
7. `.gitignore` specifies files and folders that we should not commit to Git, such as `node_modules`.

## Base

>**Hint: View changes in browser while coding**
>
>Run app with `npm start` to view latest changes in browser while writing app logic. If you haven't already, install packages with `npm install` from the root of the repo before running `npm run dev`.

### Mix JavaScript and HTML syntax in JSX

`src/App.jsx` contains the `App Component`, the root React element, in our React app and we will write our clock app logic there.&#x20;

Notice the starter code in `App.jsx` contains mostly HTML. Let's add JavaScript to it to render a date. Replace the `return` statement with the following code.

```jsx
  return (
    <>
      <div>
        <img src={logo} className="logo" alt="Rocket logo" />
      </div>
      <h1>World Clock</h1>
      <div className="card">
        <p>
          Edit <code>src/App.jsx</code> and save to test HMR
        </p>
        <p>{new Date().toString()}</p>
      </div>
    </>
  );
```

Notice we have declared a new <a href="https://www.w3schools.com/js/js_dates.asp" target="_blank">JavaScript Date object</a> from within the `<p>` tags and told React to render that date as a string in our UI. This is possible because React supports JSX syntax that enables mixing of HTML and JavaScript. Feel free to add more JavaScript elements in JSX with curly braces `{}` like we did with the date.

### Render current date and time every second

`src/main.jsx` is where we render the root React element of our React app. Let's observe how we can update it to render the current date and time every second to simulate a clock.

Comment out the code that renders `App` in `main.jsx` and add the following `tick` function and `setInterval` function call below it. Save the file to observe changes in the browser.

```jsx
{/*
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
*/}

function tick() {
  const element = (
    <div>
      <h1>Hello, world!</h1>
      <h2>It is {new Date().toLocaleTimeString()}.</h2>
    </div>
  );
  root.render(element);
}

setInterval(tick, 1000);

const root = ReactDOM.createRoot(document.getElementById("root"));

```

Notice our app now displays a digital timestamp that updates every second within the browser.

`setInterval` calls the function in its 1st parameter at the interval specified by the number of milliseconds in its 2nd parameter. In this case it is re-assigning the data and time via the `new Date()` method and re-rendering the UI every second.&#x20;

We should also consider how we have altered the original render method. We assign the `ReactDom.createRoot(document.getElementById('root'))` to a variable named root, such that we can call the render method within the `tick` function.&#x20;

Notice the value of `element` in `tick` looks similar to the element returned in the `return` statement within the `App.jsx`, that being, HTML with a JavaScript date inserted using curly braces `{}` in JSX.

The difference between `main.jsx` and `App.jsx` is the `setInterval` in `main.jsx` that calls `tick` and hence `root.render` method every second.&#x20;

## Comfortable (Second day)

### `setInterval` inside the `App` component


>Requires students to have reviewed Components and Props, React Hooks State and Lifecycle sections of React docs

What if our app had more UI elements than just the clock and we did not want to re-render the root element in `main.jsx` every second? Re-rendering the root element would be costly because it would re-render all components in our app, not just the clock.

Luckily components have lifecycle methods that run only on component mount, such that we can call `setInterval` from within a clock-specific React component without re-rendering the root element. We can achieve this using the useEffect hook. We will now move our clock logic into the `App` component so we do not need to modify the way we render our root element in `main.jsx`.

#### Undo changes in `main.jsx`

Undo the changes we made to `main.jsx` in the previous section. Our `main.jsx` file should now contain the following code below imports.

```jsx
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
```

#### Add state to `App` component to store date and time in local state

Add state to our `App` component that will store the date and time that we will update every second in a state variable named `date`. Add the `date` state using the useState method from React, instead of calling `new Date()` in our `return` statement, update the JSX to read date from `date` variable instead. `date` will contain the current value of the `date` variable in component state. When calling the `useState` function pass new Date() as the intial value. Our `App` component should look like the following.

```jsx
import logo from "/logo.png";
import "./App.css";
import {useState} from 'react'

export default function App() {
  const [date, setDate] = useState(new Date())
  return (
    <>
      <div>
        <img src={logo} className="logo" alt="Rocket logo" />
      </div>
      <h1>World Clock</h1>
      <div className="card">
        {/* Render date value that is stored in state */}
        <p>{date.toString()}</p>
      </div>
    </>
  );
}
```

You may notice that the date is still static and doesn't change at this stage. We will now add the `setInterval` code to allow our clock to re-render every second from within the `App` component.

#### Add Component useEffect method to update the date state every second and teardown state-updating logic when Component unmount's

Import the `useEffect` hook from `React`,  we can implement this method above the `return` statement inside the `App` Component in `App.jsx`. Add a `setInterval` function call inside the `useEffect` method that updates the `date` variable in local state to a new date with `setDate` every second. Save the timer ID returned by `setInterval` variable such as `timerId`, and call `clearInterval` on that `timerId` within the teardown function that can be implemented in useEffect.

The clock in our UI should now automatically update every second! <a href="https://github.com/rocketacademy/world-clock-3.2" target="_blank">Here</a> is a reference solution for this section.

>**Full reference solution at bottom of page**
>
>Rocket exercises will typically have a reference solution at the bottom of each exercise page. We will provide code examples inline for explanation, but otherwise we hope you will attempt the exercises on your own and review reference solutions afterward.

### Refactor clock display logic into its own Component

Imagine now that we wish to render multiple clocks to represent different time zones. We would like to do this inside `App` because `App` is the root Component.

Naively we could copy-paste our previous clock logic multiple times in `App` to achieve this, changing the parameters passed to `new Date()` in our `setInterval` callback function to set each date to a different timezone. This would cause much repeated code, violate the <a href="https://en.wikipedia.org/wiki/Don't_repeat_yourself" target="_blank">DRY (don't repeat yourself) principle</a>, leading to increased chances of bugs in our code.

A better solution would be to encapsulate all clock logic in a new Component called `Clock`, and use the `Clock` Component multiples times in `App`, passing timezone information as a prop to each `Clock` Component for it to render the date and time of the relevant timezone.

#### Move clock logic into `Clock` Component

Create a new file `Clock.jsx` inside the `src` folder. Define a new React Component `Clock` inside (feel free to mimic the structure of `App.jsx`) with all clock-related logic from `App.jsx`. Remember to remove the `App`-specific HTML tags in the `return` statement (everything other than the `p` tags with date string), and to `export default` the `Clock` Component at the bottom of the file, or during function declaration.

#### Remove clock logic from `App` Component

Now that our clock logic is in `Clock.jsx`, remove all clock-related logic from `App.jsx` such that the `App` Component now only contains the `return` statement with two `div`s and a `h1` tag, note that the first `div` contains an image.

#### Import `Clock` and use it in `App`

Import our `Clock` Component from `App.jsx` with code like the following below the other imports in `App.jsx`.

```jsx
import Clock from "./Clock.jsx";
```

Use the `Clock` Component in the `return` statement of `App` where we used to have our `p` tags. For now, add a single `Clock` instance and verify that our clock works when we run our app. Our `return` statement might look like the following.

```jsx
  return (
   <>
      <div>
        <img src={logo} className="logo" alt="Rocket logo" />
      </div>
      <h1>World Clock</h1>
      <div className="card">
        <Clock />
      </div>
    </>
  );
```

We now have a clock that we can re-use in anywhere in our app with a single line of code! <a href="https://github.com/rocketacademy/world-clock-3.2/commit/7d5d46e54968058770bfd720ce97a0f86b5ba4f1" target="_blank">Here</a> is a reference solution for this section

### Add timezone data to `Clock` via props

We will now add multiple clocks, each with a different time zone. Because clock logic for different time zones is the same except for time zone specification, we would not want to create a separate `Clock` Component for each time zone. Instead, we will modify `Clock` to accept time zone as a prop and render time according to the specified time zone. We will then declare multiple `Clock`s with different time zones in `App`.

#### Research how to display different time zones with JavaScript Dates

JavaScript Dates do not store dates in a specific time zone, but are able to render the date that they store in any time zone with a built-in <a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Date/toLocaleString" target="_blank">`toLocaleString`</a> method. We can use `toLocaleString` like the following.

```javascript
const date = new Date();
date.toLocaleString('en-GB', { timeZone: 'Asia/Singapore' })
```

The first parameter to `toLocaleString` is a language code (see all legal language codes <a href="https://www.w3schools.com/jsref/jsref_tolocalestring.asp#parameter-values" target="_blank">here</a>), and the 2nd parameter is an options object that allows us to specify time zone (see all valid time zones <a href="https://en.wikipedia.org/wiki/List_of_tz_database_time_zones#List" target="_blank">here</a> in "TZ database name" column).

Now that we know how to render a date in a specific time zone, we can accept a time zone string as a prop and use it to customise `Clock`!

#### Update rendered date string in `Clock` to use `toLocaleString` with time zone prop

Update the `return` statement in our `Clock` Component to render the date with `toLocaleString` instead of `toString`. Pass a language code (whichever you prefer) and time zone option as parameters to `toLocaleString`, where the time zone comes from props via `props.timeZone`.

In `App.jsx`, update our `App` component to render 3 clocks, each with a different time zone. Specify time zones with props like in the below code snippet. Feel free to pick whichever time zones are most relevant to you!

```jsx
<Clock timeZone="Asia/Singapore" />
```

#### Add time zone label to each clock

To make it clearer which time zone each clock is rendering, add a time zone label next to the date string in `Clock`'s `return` statement. This can be any string that represents the time zone.

Great job on making a clock app that shows multiple time zones! <a href="https://github.com/rocketacademy/world-clock-3.2/tree/clock_component_prop/src" target="_blank">Here </a>is a reference solution for this section. Don't forget to review the reference solution at the bottom of the page to see how Rocket implemented our full app. Coding is like writing an essay and there are many right answers, so don't fret if yours looks different.

## Improve clock UI with React Bootstrap grid system

Render time zone and time in separate columns with <a href="https://react-bootstrap.github.io/docs/layout/grid" target="_blank">React Bootstrap's grid system</a> to make our clock information easier to parse. We may wish to implement the grid system and time zone labels in `App.jsx` such that our `Clock` component can just render the time for the relevant time zone.

The grid might look something like the following.

| City        | Clock                                      |
| ----------- | ------------------------------------------ |
| Los Angeles | `<Clock timeZone="America/Los_Angeles" />` |
| London      | `<Clock timeZone="Europe/London" />`       |
| Singapore   | `<Clock timeZone="Asia/Singapore" />`      |

## More Comfortable: `WorldClock` component with dynamic number of clocks

Refactor our world clock UI into its own component `WorldClock` in its own file such that others can use it to create world clock UIs with a custom number of clocks with a custom set of timezones. `WorldClock` should accept a `clockData` prop that is an array of time zone strings, where each string corresponds to a new clock. `WorldClock` should use the `Clock` component internally, and our `App` component should import and use `WorldClock`. You may find the upcoming reading in <a href="https://react.dev/learn/rendering-lists" target="_blank">Lists and Keys</a> helpful for mapping an array of time zone strings to `Clock` components.

## Submission

Submit a pull request to the `main` branch of Rocket's World Clock repo and share your PR link in your discussion channel.

