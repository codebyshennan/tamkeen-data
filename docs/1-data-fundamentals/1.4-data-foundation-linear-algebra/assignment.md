# Assignment: Data Foundation with Numpy

## Setup

First, import numpy and create the following arrays:

```python
import numpy as np

# Student test scores for 3 subjects (math, science, english)
scores = np.array([
    [85, 92, 78],
    [90, 88, 95],
    [75, 70, 85],
    [88, 95, 92],
    [65, 72, 68],
    [95, 88, 85],
    [78, 85, 82],
    [92, 89, 90]
])

# Student names
names = np.array(['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry'])

# Random 4x4 matrix for linear algebra operations
matrix_A = np.random.randint(1, 10, size=(4, 4))
matrix_B = np.random.randint(1, 10, size=(4, 4))
```

## Tasks

1. Array Operations and Indexing

   - Calculate the average score for each student across all subjects
   - Find the highest score in each subject
   - Select all students who scored above 90 in any subject
   - Create a boolean mask to find students who passed all subjects (passing score is 70)

2. Array Manipulation

   - Reshape the scores array to be 12x2
   - Create a new array with standardized scores (subtract mean and divide by std dev)
   - Sort the students by their average score in descending order
   - Use array methods to find min, max and mean for each subject

3. Linear Algebra

   - Multiply matrix_A and matrix_B using matrix multiplication
   - Calculate the determinant of matrix_A
   - Find the inverse of matrix_A (if it exists)
   - Calculate the eigenvalues of matrix_A

4. Advanced Operations
   - Use broadcasting to add 5 points to all math scores (first column)
   - Find unique scores across all subjects
   - Use boolean indexing to find students who scored above average in all subjects

## Expected Format

Show your work with clear explanations. For each task, your output should look like:

```python
# Task 1.1: Average scores per student
average_scores = scores.mean(axis=1)
print("Average scores:", average_scores)
print("Students and their averages:")
for name, avg in zip(names, average_scores):
    print(f"{name}: {avg:.2f}")
```

## Bonus Challenge

Create a function that takes a student's name as input and returns:

- Their individual scores
- Their ranking in each subject
- A boolean indicating if they're in the top 3 performers overall

## Deliverable

Submit your solution as a Python script with:

1. All code clearly commented with
   1. Brief explanations of your approach for complex operations
   2. Any assumptions or additional features you implemented
2. Results for each task printed with appropriate labels
