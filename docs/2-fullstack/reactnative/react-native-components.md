# React Native Components and Layout

## Learning Objectives
1. Identify the different components that React Native has out of the box
2. Use React Native components to create a simple profile page
3. Apply styling and flexbox to a React Native application

## Introduction

React Native provides a number of built-in Core Components ready for you to use in your app. They are separated into several categories such as User Interfaces and List Views. Feel free to explore the documentation into this [page](https://reactnative.dev/docs/components-and-apis#basic-components)

For this lesson, we will be taking a look at the following basic components:
- View
- Text
- Image
- TextInput
- ScrollView
- StyleSheet

## App Creation

Create a new application called `CompAndLayoutApp` using the npx create-expo-app:

```sh
npx create-expo-app --template blank CompAndLayoutApp
```

Open App.js and code the next sections there. Note that the .js format for our React Native app works very similar to how .jsx works. You can still use all the capabilities that our jsx with our App.js.

## Text and View components

The most fundamental component for building a UI, **View** is a container that supports layout with flexbox, style, some touch handling, and accessibility controls. It maps directly to the native view equivalent on whatever platform React Native is running on, whether that is a UIView, android.view, etc.

The **Text** component is a React component for displaying text. Text supports nesting, styling, and touch handling.

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

The Image component can also use images online, however, it has to be styled first before the image can be used. In order to style components, the StyleSheet component can be used.

Import the StyleSheet component from react-native:

```js
import { Text, View, Image, StyleSheet } from "react-native";
```

Create a styles object outside of the function component:

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

Then add the Image component

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

The TextInput component is a foundational component for inputting text into the app via a keyboard. Props provide configurability for several features, such as auto-correction, auto-capitalization, placeholder text, and different keyboard types, such as a numeric keypad. This is not something new to us. To have TextInput component working, we have to use a state hook `useState`.

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

function App(){
    const [text, setText] = useState(null); 
}

```

# TODO: Add ScrollView