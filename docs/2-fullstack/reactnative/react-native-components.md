# React Native Components and Layout

## Learning Objectives
1. Identify the different components that React Native has out of the box
2. Use React Native components to create a user interface
3. Recall how to use `useState` hook to get inputs from a text input
4. Apply styling and flexbox to a React Native application

## Introduction

React Native provides a number of built-in Core Components ready for you to use in your app. Components follow the same convention with React. They are separated into several categories such as User Interfaces and List Views. Feel free to explore the documentation into this [page](https://reactnative.dev/docs/components-and-apis#basic-components)

For this lesson, we will be taking a look at the following basic components:
- View - The most fundamental component for building a UI
- Text - A React component for displaying text.
- Image - A React component for displaying different types of images
- TextInput - A React component for inputting text into the app via a keyboard
- StyleSheet - A React component used to add an abstraction layer for styling
- ScrollView - A React component that is used as a wrapper to enable scrolling

## App Creation

Create a new application called `CompAndLayoutApp` using the npx create-expo-app:

```sh
npx create-expo-app --template blank CompAndLayoutApp
```

Open App.js and code the next sections there. Note that the .js format for our React Native app works very similar to how .jsx works. You can still use all the capabilities that our jsx with our App.js.

## Text and View components

The most fundamental component for building a UI, **View** is a container that supports layout with flexbox, style, some touch handling, and accessibility controls. 

It maps directly to the native view equivalent on whatever platform React Native is running on, whether that is a UIView, android.view, etc.

The **Text** component is a React component for displaying text. Text supports nesting, styling, and touch handling.

Components should be first imported from `react-native` before they can used in the app.

Let's try to use the two components in our application:

```js
//App.js
import { Text, View } from "react-native"; //Add this to import the components from react-native

...
//In our App function
return (
  <View>
    <Text>Hello, smile!</Text>
  </View>
);
```

> Note: In case you find it's font too tiny, use inline style to increase its font size: 
```js
<Text style={{fontSize:50}}>Hello, smile!</Text>
```

Try adding more Text components and see what happens.

## Image component

The Image component is a React component for displaying different types of images, including network images, static resources, temporary local images, and images from local disk, such as the camera roll.

Include the Image component in the import:

```js
import { Text, View, Image } from "react-native";
```

### Using local images

To use local images, put the image into the project's `assets` folder. Create a folder called `images` in the assets folder and store the image there. 

> Note: There are placeholder images available over the web. Just choose and download one as needed.
> Note: This assumes that you have an image called `sample.png` in the `assets/images` folder

Import the image as a variable:
```js
import samplePng from "./assets/sample.png";
```

Use the Image component in the return statement:
```js
return (
  <View>
    <Text>Hello, smile!</Text>
    <Image source={samplePng}></Image>
  </View>
);
``` 

### Using images online

The Image component can also use images online, however, it has to be styled first before the image can be used by specifying the dimensions of the image. In order to style components, the StyleSheet component can be used.

Import the StyleSheet component from react-native:

```js
import { Text, View, Image, StyleSheet } from "react-native"; //Add the StyleSheet in the import
```

Create a `styles` object outside of the function component:

```js
function App(){
    ...
}

const styles = StyleSheet.create({
  image: {
    height: 500,
    width: 500,
  },
});
```

Then add the Image component, for images from the web, provide a `style` and `source` with the `uri` property. 

```js
return (
  <View>
    <Text>Hello, smile!</Text>
    <Image
      style={styles.image}
      source={{
        uri: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRR2wcpqmDDUjViB6TEfWO-hxzaf5cwENejaQ&usqp=CAU",
      }}
    ></Image>
    <Image source={samplePng}></Image>
  </View>
);
```

See the results of adding the two images.

## TextInput and ScrollView components

The TextInput component is a foundational component for inputting text into the app via a keyboard. Props provide configurability for several features, such as auto-correction, auto-capitalization, placeholder text, and different keyboard types, such as a numeric keypad. 

This is not something new to us. To have TextInput component working, we have to use a state hook `useState`.

Import the TextInput component from react-native:

```js
import { Text, View, Image, StyleSheet, TextInput } from "react-native";
```

Include the TextComponent in the return block:

```js
return (
  <View>
    <Text>Hello, smile!</Text>
    <Image
      style={styles.image}
      source={{
        uri: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRR2wcpqmDDUjViB6TEfWO-hxzaf5cwENejaQ&usqp=CAU",
      }}
    ></Image>
    <Image source={monkeyPng}></Image>
    <TextInput
      style={styles.input}
      value={text}
      onChangeText={setText}
    ></TextInput>
  </View>
);
```

Add styling for the TextInput:
```js
const styles = StyleSheet.create({
  image: {
    height: 500,
    width: 500,
  },
  input: {
    height: 40,
    margin: 12,
    borderWidth: 1,
    padding: 10,
  },
});
```

Implement state management via useState:
```js
import React, { useState } from "react"; // the import
//other imports

function App(){
    const [text, setText] = useState(null); 
    ...
}

```

You would notice that the TextInput might not be visible. This is because that it is located at the bottom of the app which the screen doesn't show. To mitigate this, we will allow the view to be scrollable. We do this by converting the View component into using the ScrollView component.

```js
import {
  StyleSheet, Text, View, Image, TextInput, ScrollView // Add the ScrollView in our import.
} from "react-native";

//and replace the View in the JSX with ScrollView
```

After replacing the View with the ScrollView component, your app show allow scrolling and you would be able to scroll and see the TextInput component.

## Flexbox

In this part, we will only use the View component and Stylesheet. This example is taken from the [official React Native Flexbox documentation](https://reactnative.dev/docs/flexbox#flex).

Create a new React Native app called flexboxApp and import the necessary components:

```js
import {View, StyleSheet} from "react-native";

function App (){
  return (
    <View
      style={[
        styles.container,
        {
          // Try setting `flexDirection` to `"row"`.
          flexDirection: "column",
        },
      ]}
    >
      <View style={{ flex: 1, backgroundColor: "red" }} />
      <View style={{ flex: 2, backgroundColor: "darkorange" }} />
      <View style={{ flex: 4, backgroundColor: "green" }} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
  },
});

export default App
```

Try playing around with the number of views and flex values. Do remember to check the documentation for additional flex properties.