# React Native Navigation

## Learning Objectives
1. Know what Navigation Containers, Navigators and Screens are.
2. Implement Tab-based and Drawer-based navigations

## Introduction

In a web browser, you can navigate to different pages using an **anchor (<a>) tag**. When the user clicks on a link, the URL is pushed to the browser history stack. When the user presses the back button, the browser gets the URL from the top of the stack, so the active page is now the previously visited page. 

However, React Native doesn't have global history stack like a web browser does -- this is where [React Navigation](https://reactnavigation.org/docs/hello-react-navigation) comes in. 

It's **native stack navigator** provides a way for your app to transition between screens and manage navigation history. 

A key difference with React Navigation is the stack navigator provides the gestures and animations that you have on your mobile devices when navigating between routes in the stack.

Feel free to explore the **Fundamentals** section and look at the **Guide** section to see examples of implementation.

For this lesson, we will be looking at two navigation implementations, the *tab-based* and *drawer-based* navigations. 

The two most common ways of navigating in mobile apps is through the use of *tab-based* and *drawer-based* navigation.Tab-based navigation puts tabs on the bottom of the screen or on the top below the header. While drawer-based navigation provides a drawer on the left or right side of the screen that is usually hidden, but can be pulled for navigating between screens.

## Tab-based Navigation

The most common style of navigation in mobile apps is **tab-based navigation**. This can be tabs on the bottom of the screen or on the top below the header (or even instead of a header). The example is taken from this [guide](https://reactnavigation.org/docs/tab-based-navigation).

### Setup

Install the `bottom-tabs` page from `react-navigation`. 

```sh
npm install @react-navigation/native @react-navigation/bottom-tabs
```

The `react-navigation/native` package allows us to use the stack navigator for our application, and the `react-navigation/bottom-tabs` allow us to create bottom tabs

*Note: You may be required to install additional dependencies on expo. You don't have to run the following commands if you are not prompted to:*

```sh
npm install react-native-screens react-native-safe-area-context
```

### App Creation

Create a React Native app using the blank template and call the app "TabNavigation".

```sh
npx create-expo-app --template blank TabNavigation
```

### Screens Creation

How tab navigation works is that you have screens that would correspond to components and when you click on the navigation, it would call on the necessary screen.

Begin by creating two screens, HomeScreen and SettingsScreen:

```js
import * as React from 'react';
import { Text, View } from 'react-native';

function HomeScreen() {
  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <Text>Home!</Text>
    </View>
  );
}

function SettingsScreen() {
  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <Text>Settings!</Text>
    </View>
  );
}
```

The styling here uses the flexbox layout, you can check out flexbox in this [link](https://css-tricks.com/snippets/css/a-guide-to-flexbox/)

### Creation of Bottom Tabs

After creating the screens, we now create the container and the bottom tabs to navigate through the different screens.

Import the `NavigationContainer` and `createBottomTabNavigator` from `react-navigation/native` and `react-navigation/bottom-tabs`.

After which, a `Tab` component is created and we use it for the screens.

```js
import * as React from 'react';
import { Text, View } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

function HomeScreen() {
  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <Text>Home!</Text>
    </View>
  );
}

function SettingsScreen() {
  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <Text>Settings!</Text>
    </View>
  );
}

const Tab = createBottomTabNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Tab.Navigator>
        <Tab.Screen name="Home" component={HomeScreen} />
        <Tab.Screen name="Settings" component={SettingsScreen} />
      </Tab.Navigator>
    </NavigationContainer>
  );
}
```

The `NavigationContainer` serves as a wrapper for the Tab navigation. The `Tab.Navigator` is the bottom tab navigation and `Tab.Screen` corresponds to each of the screen components.


Try playing around with the tab navigation by creating more screen components.

## Drawer Navigation

### Setup

For this example, we will be using the React Navigation's drawer dependency. You can check out more examples here in the [link](https://reactnavigation.org/docs/drawer-based-navigation)

First, create a new RN app via the `npx create-expo-app` command:

```sh
npx create-expo-app --template blank DrawerNavigation
```

Install the dependencies via the `npm install` command:

```sh
npm install @react-navigation/native @react-navigation/drawer react-native-screens react-native-safe-area-context react-native-gesture-handler react-native-reanimated@3.10.1
```

The `react-navigation/native` package allows us to use the stack navigator for our application, and the `react-navigation/drawer` allow us to create a drawer. We also need to use react-native-gesture-handler to allow us to use swiping.

### Screens Creation

Drawer screens work similarly on how it works for Tab screens.

Begin by creating two screens, Feed and Article:

```js
import * as React from 'react';
import { Text, View } from 'react-native';

function Feed() {
  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <Text>Feed!</Text>
    </View>
  );
}

function Article() {
  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <Text>Article!</Text>
    </View>
  );
}
```

### Creation of the Drawer

After creating the screens, we now create the container and the drawer to navigate through the different screens.

Import the dependencies as seen in the code below:

```js
import * as React from 'react';
import { Text, View } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createDrawerNavigator } from '@react-navigation/drawer';

function Feed() {
  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <Text>Feed!</Text>
    </View>
  );
}

function Article() {
  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <Text>Article!</Text>
    </View>
  );
}

const Drawer = createDrawerNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Drawer.Navigator>
        <Drawer.Screen name="Feed" component={Feed} />
        <Drawer.Screen name="Article" component={Article} />
      </Drawer.Navigator>
    </NavigationContainer>
  );
}
```

The `NavigationContainer` serves as a wrapper for the Tab navigation. The `Drawer.Navigator` is the drawer navigation and `Drawer.Screen` corresponds to each of the screen components.


Try playing around with the drawer navigation by creating more screen components.


There are many other ways to navigate and pass information to different screen. Try to experiment with the additional [features](https://reactnavigation.org/docs/getting-started) for React Navigation.

## Exercise

### Code Along
Make both the `TabNavigation` and `DrawerNavigation` examples work.