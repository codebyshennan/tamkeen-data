# Nodemon

## Learning Objectives

1. Demonstrate how to install Nodemon globally using npm
2. Show how to use Nodemon to automatically restart a Node.js application when file changes are detected

## Introduction

[Nodemon](https://www.npmjs.com/package/nodemon) is an application that restarts our Node app every time we change a file that the app depends on. This is especially useful in development when we make frequent changes to our code. Without Nodemon we would need to manually quit and restart our app on code changes.

## Usage

1.  Install Nodemon globally to run Nodemon from all folders.

    ```
    npm i -g nodemon
    ```
2.  Run `nodemon` on the entry file of our app. When any app files changes, Nodemon will restart the app.

    ```
    nodemon index.js
    ```
