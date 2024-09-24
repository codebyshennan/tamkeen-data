# React Native External Libraries

## Learning Objectives

1. Install Expo SDK packages
2. Configure Expo plugins in app.json file
3. Create a demo barcode scanner mobile app

## Expo SDK

The Expo SDK provides access to the device's system functionalities such as contacts, camera, gyroscope, GPS location, etc. in the form of packages. 


You can install any Expo SDK package using the `npx expo install` command. After installing the packages, you can import them into the code. This allows you to use the package's function and components.


Some packages require additional configuration and permissions in the `app.json` file before it can use the device's functionalities.

## Sample SDK - Phone Camera

`expo-camera` provides a React component that renders a preview of the device's front or back camera. 


The camera's parameters such as zoom, torch, and flash mode are adjustable. Using `CameraView`, you can take photos and record videos that are saved to the app's cache. The component is also capable of detecting bar codes appearing in the preview.


For this lesson, we will be using it to do simple barcode scanning. You can check on the documentation of Expo Camera [here](https://docs.expo.dev/versions/latest/sdk/camera/)

### Setup

Create a new expo app using the `npx create-expo-app` command:

```sh
npx create-expo-app --template blank CameraDemoApp
```

Install the expo-camera library using the `npx expo install` command:

```sh
npx expo install expo-camera
```

Configure the `app.json` to allow permissions for the camera:

```json
//app.json
//Add the part for the plugins
{
  "expo": {
    "plugins": [
        [
            "expo-camera",
            {
                "cameraPermission": "Allow $(PRODUCT_NAME) to access camera."
            }
        ]
    ],
    "name": "CameraDemoApp",
    "slug": "CameraDemoApp",
...
  }
}
```

What will happen is that when the app launches, it will ask for permission for camera use. 

After setting up the configuration in `app.json`, we code the content for `App.js`.

```js
//App.js
import { StyleSheet, Text, View, Button } from 'react-native';
import { CameraView, Camera } from 'expo-camera';
import React, { useState, useEffect } from 'react';

export default function App() {
  
  // Set component states here
  const [hasPermission, setHasPermission] = useState(null);
  const [scanned, setScanned] = useState(false);

  // Request for permission to access the user's camera when component loads
  useEffect(() => {
    const getCameraPermissions = async () => { 
        const { status } = await Camera.requestCameraPermissionsAsync();
        setHasPermission(status === 'granted');
    };

    getCameraPermissions();
  }, []);

  // Bar code scanner handler - displays the data type and its content
  const handleBarCodeScanned = ({ type, data }) => {
    setScanned(true);
    alert(`Bar code with type ${type} and data ${data} has been scanned!`);
  };

  // Checks component state for camera access permission
  if (hasPermission === null) {
    return <Text>Requesting for camera permission</Text>;
  }
  if (hasPermission === false) {
    return <Text>No access to camera</Text>;
  }

  // JSX starts here
  return (
    <View style={styles.container}>
      <CameraView 
        onBarcodeScanned={scanned ? undefined : handleBarCodeScanned}
        style={StyleSheet.absoluteFillObject}
        barcodeScannerSettings={{ //This property allows you to provide what kinds of barcodes/qr codes it will scan.
          barcodeTypes: ["qr", "pdf417", "code128"],
        }}
      />
      {scanned && <Button title={'Tap to Scan Again'} onPress={() => setScanned(false)} />}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
  },
});
```

The useEffect hook is used to ask for permissions for the camera use, it uses the `requestCameraPermissionsAsync` function to request access for the phone's camera.


The handler function `handleBarCodeScanned` handles the barcode scanning based on the event `onBarcodeScanned`.

### Testing

The barcode settings can be adjusted to handle what kinds of barcodes the application can scan.

You can test out the app by creating barcodes in this [website](https://barcode.tec-it.com/en). You can choose *Code-128* under Linear Codes and *QR* under 2D Codes.

### Exercise

Create a Code-128 barcode with Data: "I love coding!" and try to scan the created barcode.