# Firebase Realtime Database

## Learning Objectives

1. Know what is Firebase Realtime Database and what it is used for
2. Set up Firebase Realtime Database with React Apps powered by Vitejs

### Introduction

{% include youtube.html id="U5aeM5dvUpA" %}
Firebase Realtime Database is a convenient tool to develop app prototypes. Source: Firebase


<a href="https://firebase.google.com/docs/database" target="_blank">Firebase Realtime Database</a> is a "NoSQL" (i.e. non-relational) JSON database that allows us to persist app data with API calls and sync that data in real-time across multiple app instances (e.g. multiple users). Realtime Database is Firebase's first product and one of their most popular.

Start by reading the following official Firebase Realtime Database tutorials linked below. We will skip the Get Started page and come back to it when working on our first exercise.

### <a href="https://firebase.google.com/docs/database/web/structure-data" target="_blank">Structure your Database</a>

Read the following Firebase official tutorial on how to structure data in Realtime Database.

1. We can think of JSON objects as JavaScript objects
2. We will learn SQL when we learn about backend
3. We will follow the structure of data in this example to build chat functionality in this module's exercises

### <a href="https://firebase.google.com/docs/database/web/read-and-write" target="_blank">Read and Write Data on the web</a>

Read the following tutorial on how to read and write data to Firebase Realtime Database. Try to understand the code in each example. We will understand it in more detail when we work on this module's exercises.

1. Skip the "(Optional) Prototype and test with Firebase Local Emulator Suite" section. We can use this later when we need it.
2. We will be using the functions mentioned in the docs to write, read, update and delete data.
3. We may use `off()` to detach listeners when our React components unmount; no need to pay too much attention to this for now.
4. We can ignore "Save data as transactions" section for now. We can revisit this when we need this functionality.

### <a href="https://firebase.google.com/docs/database/web/lists-of-data" target="_blank">Work with Lists of Data</a>

Read the following tutorial on how to read to, write to, sort and filter lists in Realtime Database.

1. We will use functionality from both "Reading and writing lists" section and "Sorting and filtering data" section in our apps.



### Sample Implementation Legacy (Class Based)

{% include youtube.html id="ffj53HLE4Yk" %}
What is Firebase


{% include youtube.html id="-xa1ec52y5Q" %}
Firebase Realtime Database


{% include youtube.html id="GbsWh2Dif_M" %}
Firebase Realtime Database Setting Data


{% include youtube.html id="p3OSy0oOlXE" %}
Firebase Realtime Database Using Data


Please checkout the finished code in this <a href="https://github.com/rocketacademy/firebase-examples-3.2" target="_blank">repository</a>, ensure that you're on the `realtimedatabase` branch. If you want to test out the application on your machine you will need to have registered an Application on Firebase with Realtime Database activated. Use the `sample.env` within the application to create an `.env` file with your Firebase credentials. With this in mind if you want to run the application on your machine you will need to install the dependencies with the command `npm install` after the installation you can then run the application with the command `npm run dev`. Then open a browser of your choice and navigate to  http://localhost:5173.
