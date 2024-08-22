# Firebase Authentication

## Learning Objectives

1. Use Firebase Authentication to authenticate our users with API calls instead of custom password-hashing or OAuth logic
2. Learn how to set up Firebase Authentication with React

### Introduction

{% include youtube.html id="8sGY55yxicA" %}
Firebase Authentication is a plug-and-play authentication solution for our apps. Source: Firebase


<a href="https://firebase.google.com/docs/auth" target="_blank">Firebase Authentication</a> allows us to authenticate our apps' users, manage their accounts in the Firebase console and easily access user auth information from within our application. It makes authentication easy so we can focus on our app logic.

Start by reading the official Firebase Authentication tutorials linked below. Skip "Sign in with a pre-built UI" and "Get Started"; we will not use the former because it is currently incompatible with React 18, and we will go through the latter in the upcoming exercise.

### <a href="https://firebase.google.com/docs/auth/users" target="_blank">Users in Firebase Projects</a>

1. We will primarily use email and password auth for learning. Feel free to use other forms of auth in your exercises and projects.
2. The Firebase Auth instances provides convenient access to the currently logged-in user. We can use this throughout our app.
3. Auth listeners provide a convenient way to trigger relevant components to re-render when a user logs in or out.

### <a href="https://firebase.google.com/docs/auth/web/manage-users" target="_blank">Manage Users</a>

1. We will almost certainly want to retrieve current user from the Auth object in our apps, e.g. to display logged-in user in places such as navbars, profile pages and comment bars.

### <a href="https://firebase.google.com/docs/rules/basics?authuser=0\&hl=en#cloud-storage" target="_blank">Firebase Storage Basic security rules</a>

1. When you have successfully setup firebase authentication you can update your security rules for Firebase storage.

