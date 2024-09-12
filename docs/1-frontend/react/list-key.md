# List and Keys

## Learning Objectives

1. Explain the purpose and importance of keys when rendering lists of JSX elements in React.
2. Implement unique and stable keys for list items to optimize React's rendering performance.
3. Evaluate different methods for generating unique keys, including database IDs, incrementing counters, and third-party libraries like UUID.

## <a href="https://react.dev/learn/rendering-lists" target="_blank">Lists and Keys</a>

JSX keys in an array help us to identify unique items from thier siblings. Keys help React to identify different JSX elements throughout their lifetimes. You can generate key values in whatever way you would like, but if you've pulled data out of a database, they might already have identities, such as an Id property. You could create your own incrementing counter to set unique keys within an application or implement <a href="https://www.npmjs.com/package/uuid" target="_blank">uuid</a> or another package that can assign unique id's within an application.

1. Always use keys when rendering JSX elements in a list for performance reasons.
2. Keys should be unique among siblings
3. Keys cannot change
