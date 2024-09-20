# Introduction to React Native

## Learning Objectives
1. Create a simple React Native App
2. Run the React Native App using an emulator and host it in the mobile phone

## Introduction

React Native projects are built on Expo development framework that simplifies the development of mobile React Native applications. Expo Go is a client used for testing React Native apps on your mobile device and is the fastest way to get up and running. It allows you to open up apps served through the Expo CLI and run your projects on the Expo Go client app on your Android/iOS device. More details about Expo Go can be found [here](https://docs.expo.dev/get-started/set-up-your-environment/).

## Software checks

Expo requires a Node.js JS runtime. To check your node version, type `node -v` on your terminal.

You should have at least **Node version 18** for Expo to work properly.

To load the app in your mobile, install the [Expo Go App](https://expo.dev/go) on your mobile phone. This app is compatible for both Android and iOS devices.

## React Native App creation

The following steps are taken from the official [environment setup guide](https://reactnative.dev/docs/environment-setup).

1. Create the React Native project using the `npx create-expo-app` command:
```sh
npx create-expo-app --template blank ReactNativeDemo
```

The blank template is a pure JavaScript template without any navigation, you can explore other templates here in this [link](https://galaxies.dev/quickwin/list-of-react-native-expo-templates).


For this lesson, we name our app `ReactNativeDemo`, but you can use any app name as needed.

2. After creating the app, you can start the development server by going to the project directory and typing the `npx expo start` command:

```sh
cd ReactNativeDemo
npx expo start
```

After starting the app, you would see a QR code that would be generated. This QR code can be scanned using the Expo Go app to load the app in your mobile device.  

3. Open Expo Go app and scan the QR code. It would bundle and launch the application. You should be able to see the text **"Open up App.js to start working on your app!"**.

> Note: If you get a "Network response timed out" error message, you might try re-connecting with a tunnel option. Press `CTRL-C` to exit the server and restart with the tunnel command: `npx expo start --tunnel`

## Android Studio Emulation

As of now, we only load our applications using our own mobile devices. However, if we are to test using different devices, we can use an emulator. Developing with React Native using emulator brings about the following benefits:

- It allows you to test multiple OS versions.
- It allows you to mock device-specific data, such as your current location.
- You don't have to use your real device.
- The power of being able to take screenshots and share screen with your team.

We will be installing Android Studio and from there, we will setup a virtual device for emulation.

### Installing Android Studio

Go to official Android Studio [download page](https://developer.android.com/studio/install), launch the installer package and follow the steps on screen.

If you are prompted that Java is required, download it at [here](https://openjdk.org/)

### Setting up the AVD (Android Virtual Device)

> Note for Linux and MacOS, please modify the path variables following the instructions in this [link](https://docs.expo.dev/workflow/android-studio-emulator/)

To create a virtual device, in the main screen of Android Studio, choose **"More Actions"** and select **"Virtual Device Manager"**, then click on the "+" button to "Create Virtual Device". 

> For standardization, select *Pixel XL* as device, and choose the OS *UpsideDownCake*.

Once setup is done, you should see a new device being displayed on your Android Virtual Device Manager. To launch the emulator, click on Launch button (denoted by the play or > button). 

Wait for emulator to finish loading up and you should see a mobile device on your screen.

### Launching the app in the emulator

To launch the app in the emulator, restart the app using the `npx expo start` command. You would see a new option called "Run on Android Device/Emulator". Select that option and wait for the emulator to download Expo Go and load the application.

## Exercise

### Setup completion

Make sure that the React Native Demo app runs on both your mobile device and the emulated device