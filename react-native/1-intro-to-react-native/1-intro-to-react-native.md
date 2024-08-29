# Introduction to React Native

## Learning Objectives

* Understand what React Native is and what it's used for
* Create React Native applications
* Install Expo CLI and Expo App to run React Native applications
* Use Android Studio to run emulators for mobile phones

## What is React Native and Expo?

React Native is a cross-platform library that builds native mobile apps (iOS and Android) using JavaScript and React. 


React Native projects are built on Expo development framework that simplifies the development of mobile React Native applications. 


Expo Go is a client used for testing React Native apps on your mobile device and is the fastest way to get up and running. It allows you to open up apps served through the Expo CLI and run your projects on the Expo Go client app on your Android/iOS device. More details about Expo Go can be found [here](https://docs.expo.dev/get-started/expo-go/).


In order to use Expo, it needs a Node runtime. To check your node version, type `node -v` on your terminal.


```sh
node -v //should reflect your current node version
```

## Creating a React Native App

Similar to creating React Apps, React Native apps can be create using the `npx create-expo-app` command in the Terminal with the format: `npx create-expo-app <app_name>`.

```sh
npx create-expo-app --template blank ReactNativeDemo
```

There are templates available for React Native that uses TypeScript. The `--template` option can be used to choose among the templates. Check out this [link](https://docs.expo.dev/more/create-expo/#--template) for additional information.


Note: For the lessons on React Native, the blank template will be used for learners to quickly spin up an app.

## Running the app

React Native apps can be started using the `npx expo start` command`

```sh
npx expo start
```

After the app is run, there will be a QR code generated for the application. To launch the app in a mobile phone, [Expo Go](https://expo.dev/go) must be installed and the QR code must be scanned using Expo Go's QR code scanner.

*Note: There are times that a `Network Response Timed Out` error message might appear. To solve this, use the tunnel option while running the app: `npx expo start --tunnel`* 

## Android Studio

Another way of loading the React Native app is through the use of an emulator.


Using an emulator brings about these benefits over loading the app in a phone.
* It allows you to test multiple OS versions.
* It allows you to mock device-specific data, such as your current location.
* You don't have to use your real device.
* The power of being able to take screenshots and share screen with your teammates

[Android Studio](https://developer.android.com/studio/install) is one such emulator that allows you to simulate devices on computers.

### Setting up Android Studio

After installing Android Studio, a SDK needs to be setup. Following the instructions in this [page](https://docs.expo.dev/workflow/android-studio-emulator/#set-up-android-studio)


After the SDK has been setup, Go to `Virtual Device Manager` (found in the `More Actions`) and create a new device by clicking the `+` icon. 


For standardization, select **Pixel XL** as device, and choose the latest OS. It is **UpsideDownCake** on the date of writing. 


To run the created device, hit the Play (">") button. Wait for android emulator to finish loading up and you should see a simulated phone on your screen.

### Launching Expo Go in Android Emulator

Go back to the project created in your lesson, start it with `npx expo start`. You would see an option `Run on Android Device/Emulator`, press the key and wait for the app to be deployed on the android emulator.