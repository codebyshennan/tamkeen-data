# React Native Components

## Learning Objectives

* Be familiarized with the different React Native Components
* Use the different React Native Components to create a user interface
* Recall how to use `useState` hook to get inputs from a text input

## Introduction

Similar to React, React Native has its own set of compoentes that are ready to use out of the box. Components follow the same convention with React. 

Here are some of the components that are used by React Native:
* View - The most fundamental component for building a UI
* Text - A React component for displaying text.
* Image - A React component for displaying different types of images
* TextInput - A React component for inputting text into the app via a keyboard
* StyleSheet - A React component used to add an abstraction layer for styling
* ScrollView - A React component that is used as a wrapper to enable scrolling

For a full list of components, feel free to check on the documentation [here](https://reactnative.dev/docs/components-and-apis).

## Pre-work

Create a new Expo App called `CompAndLayoutApp` using the npx command:

```sh
npx create-expo-app --template blank CompAndLayoutApp
```

## View and Text Components

The View component is the most fundamental component for building a UI, it is a container that supports layout with flexbox, style, some touch handling, and accessibility controls. 


View maps directly to the native view equivalent on whatever platform React Native is running on, whether that is a UIView, android.view, etc.


The Text component is a React component for displaying text. Text supports nesting, styling, and touch handling.


Components should be first imported from `react-native` before they can used in the app.

```js
//App.js
import { Text, View } from "react-native"; //This line imports the Text and View components
...

return (
  <View>
    <Text>Hello, smile!</Text>
  </View>
);
```

Note: The font size might be small for certain screens. To fix this, use inline style to increase the font size: 

```js
<Text style={{fontSize:50}}>Hello, smile!</Text>
```

## Image Component

The Image component is used for displaying images, including network images, static resources, temporary local images, and images from local disk, such as the camera roll.

### Adding local images

First, include the Image component in the import:

```js
import { Text, View, Image } from "react-native";
```

Import the image as a variable (don't forget to download a sample image and put the image in the assets folder) and call the Image component in JSX.

```js
...
import sampleImg from "./assets/sampleImg.png";
...
return (
  <View>
    <Text>Hello, smile!</Text>
    <Image source={sampleImg}></Image> {/*place the component here*/}
  </View>
);
```

### Adding images from the web

*Note that for network and data images, you will need to manually specify the dimensions of your image!*

For images from the web, provide a style and source with the uri property. 

```js
import { Text, View, Image, StyleSheet } from "react-native"; //Add the StyleSheet in the import
...
return (
  <View>
    <Text>Hello, smile!</Text>
    <Image
      style={styles.image}
      source={{
        uri: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRR2wcpqmDDUjViB6TEfWO-hxzaf5cwENejaQ&usqp=CAU",
      }}
    ></Image>
    <Image source={sampleImg}></Image>
  </View>
);
...
//Add the style outside the App function
const styles = StyleSheet.create({
  image: {
    height: 500,
    width: 500,
  },
});
```

## TextInput and ScrollView Components

The TextInput component is used for inputting text into the app via a keyboard. Props provide configurability for several features, such as auto-correction, auto-capitalization, placeholder text, and different keyboard types, such as a numeric keypad.

In order to have TextInput component working, the `useState` hook is needed.

Add the TextInput component, as well as the useState component in the imports

```js
import {useState} from "react";
import { Text, View, Image, TextInput } from "react-native";
...
const [text, setText] = useState(null);

return (
  <View>
    <Text>Hello, smile!</Text>
    <Image
      style={styles.image}
      source={{
        uri: "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRR2wcpqmDDUjViB6TEfWO-hxzaf5cwENejaQ&usqp=CAU",
      }}
    ></Image>
    <Image source={sampleImg}></Image>
    <TextInput
      style={styles.input}
      value={text}
      onChangeText={setText}
    ></TextInput>
  </View>
);
...
//Add styling after App function
const styles = StyleSheet.create({
  input: {
    height: 40,
    margin: 12,
    borderWidth: 1,
    padding: 10,
  },
});
```

At this point, you would notice that certain components are no longer visible as they are pushed down to outside the screen. In order for us to see the components outside of the screen, we will make the screen scrollable using the `ScrollView` component.

The `ScrollView` component allows screen scrolling for us to see components that are out of the screen view.

Replace the `View` component with `ScrollView` and see the effect on the screen.