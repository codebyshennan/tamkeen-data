# "Hello World" in React!

## Learning Objectives

- Create a new React application using Vite and explain its key file structure.

- Demonstrate the use of JSX to combine JavaScript and HTML-like markup in React components.

- Apply JavaScript logic within JSX using curly braces to create dynamic content.

## <a href="https://react.dev/reference/react-dom/client/createRoot#usage" target="_blank">Hello World</a>

In this section you should begin to understand that React applications are bound to a single root, this code is generated when we start a new React application using <a href="https://vitejs.dev/guide/" target="_blank">Vitejs</a>. Give this a shot yourself and generate a new React application with <a href="https://vitejs.dev/guide/" target="_blank">Vitejs</a>, you should be able to see a similar file structure to the official documentation above.

To create a new React application using Vite, open up your CLI tool and run this command in the directory where you want to create your application:

`npm create vite@latest`

You will be prompted to input a name.

Then you will be asked what type of application you want to create, use your arrow keys to navigate to 'React' and press 'enter'.

Then using arrow keys once more navigate down to 'JavaScript', if you don't need standard build tools later in development you can choose 'JavaScript + SWC' and click enter.

Now you can follow the instructions within the CLI to first navigate into the newly created directory, then install the required dependancies and finally run the React application using the command `npm run dev`, to view the application you should open the URL that is shown in your CLI, it should be something like <a href="http://localhost:5173/" target="_blank">**http://localhost:5173**</a>, the output on the page should be similar to the image below. 

<figure><img src="../../.gitbook/assets/Screenshot 2023-10-18 at 2.49.25 PM.png" alt=""><figcaption><p>Vite and React</p></figcaption></figure>

There are some key files that you should be aware of within this starter code. Note that the index.html is in the root of the project directory, it is in the entry point to your application and is required for the React App. Please explore the files within the starter code and consider the points below,

1. Within the src/main.jsx file `document.getElementById('root')` is a DOM (Document Object Model) command that retrieves the HTML element with ID "root". React renders our app inside that element. React removes the need to write DOM code and the above command is the only DOM command we will need.
2. <mark style="color:red;">**Note**</mark> that if you want to add and use images within your Vite React applications please add the images into the `public` directory and import as usual.
3. Rocket recommends that you familiarise yourself with as many of the provided examples as you can to understand and digest the relevant concepts
4. Rocket recommends completing most of the "Learn React" guide before attempting the "Tutorial: Tic Tac Toe" so the tutorial's concepts sink in better
5. If there is anything you do not understand in the guide, please let your SL know and we can include it in these notes!

## <a href="https://react.dev/learn/writing-markup-with-jsx" target="_blank">Introducing JSX</a>

JSX or JavaScript and XML affords developers the opportunity to write HTML-like markup within a JavaScript or JSX file. Without using JSX developers would be forced to utilise React <a href="https://react.dev/reference/react/createElement" target="_blank">createElement</a>, this is less readable and intuitive than JSX. By employing JSX we can combine rendering logic alongside markup within what developers name components. While JSX looks like HTML it is stricter and can display dynamic information. It should be noted that JSX has its own rules that should be followed.

1. JSX elements may only return a single root element, to display multiple elements just wrap them within a single parent tag.
2. Close all HTML tags.
3. DOM stands for "Document Object Model", which is a JavaScript representation of HTML rendered on a web page. Frontend frameworks like React use the DOM to programmatically manipulate UI without manually specifying HTML. Rocket recommends <a href="https://www.w3schools.com/js/js_htmldom.asp" target="_blank">W3School's intro to JavaScript HTML DOM</a> (just the 1st page) for a primer. For Rocket's Bootcamp we can stop at DOM Intro without reading W3School's subsequent pages on DOM.
4. Use camelCase for most if its props,<a href="https://react.dev/learn/writing-markup-with-jsx#3-camelcase-salls-most-of-the-things" target="_blank"> click here for more details</a>. To apply CSS classes to JSX elements we will need to use the `className` keyword instead of `class`, which we used with vanilla HTML. This is because `class` is a reserved keyword in JS used to declare classes (which we will see in 4: Components and Props below).

## <a href="https://react.dev/learn/javascript-in-jsx-with-curly-braces" target="_blank">JavaScript in JSX</a>

Within JSX, "{ }" or "curly braces" allow developers to write and execute JavaScript bound to the HTML-like code. This makes JSX more dynamic than HTML and allows for data binding basically straight out of the box.

1. When you want to add JavaScript logic or code into JSX you are able to do so by using curly braces within the JSX. Think of these using these braces { } like a window into the world of JavaScript.
2. You can pass string attributes to JSX, do this using single or double quotations.
3. You can only use curly braces as text directly within a JSX tag **OR** as attributes immediately following the `=` sign.
4. You should use double curly braces in some instances when writing JSX. Such as using inline CSS or passing a js object into a JSX element. 

React follows what we call a "declarative" UI paradigm, where we tell our computers how the UI should look, but not how to achieve that look. The declarative paradigm is a layer on top of the "imperative" paradigm of DOM manipulation more commonly used before React.
