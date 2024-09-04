# Lifting States Up

## Learning Objectives

- Implement the "lifting state up" technique to manage shared state between multiple React components.

- Analyze component hierarchies to determine the appropriate location for shared state management.

- Construct a React application that maintains a single source of truth by passing state and event handlers as props from parent to child components.

## <a href="https://react.dev/learn/sharing-state-between-components" target="_blank">Lifting State Up</a>

Managing state on components can get challenging and confusing when you need to share data between multiple components.  Especially so while still having to preform live updates and dynamically render content. The process of lifting up state means your children components actually receive data as props. Instead of storing the data on each child, you store the data on a shared parent of both child components. 

1. It is important that we pass both the state and handler function from `Accordian` to `Panel` so the app can maintain only 1 "source of truth" state for the active value in `Panel`. If it helps, Rocket recommends drawing the component hierarchy to visualise how data flows in the application and verifying our understanding with our classmates and section leader.
