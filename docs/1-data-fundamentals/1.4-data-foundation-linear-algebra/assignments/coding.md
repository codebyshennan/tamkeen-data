# Assignment: Data Foundation with Numpy

> **Submit your work on Skills Union →** <https://skillsu.com/member/assessment>

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


## Hints

<details>
<summary>Show hints</summary>

## 1. Array operations and indexing
- **Where:** [ndarray basics](../ndarray-basic.md), [ndarray methods](../ndarray-methods.md), [boolean indexing](../boolean-indexing.md).
- **Think:** Pick the right `axis`. With `scores` shaped `(students, subjects)`, **per student** means collapse subjects → `axis=1`; **per subject** means collapse students → `axis=0`.
- **Starter:**
  ```python
  per_student_avg = scores.mean(axis=1)   # one value per row
  per_subject_max = scores.max(axis=0)    # one value per column
  high_in_any   = names[(scores > 90).any(axis=1)]
  passed_all    = names[(scores >= 70).all(axis=1)]
  ```

## 2. Array manipulation
- **Where:** [ndarray basics](../ndarray-basic.md) — `reshape`, [ndarray methods](../ndarray-methods.md) — reductions and sorting.
- **Think:**
  - **Reshape to 12×2:** the product 12×2 = 24 must equal the original 8×3 = 24. Use `reshape(12, 2)`.
  - **Standardize:** "(x − mean) / std" **per subject** ⇒ both reductions use `axis=0` so broadcasting lines up the columns.
  - **Sort descending by average:** `argsort()` ascends; reverse with `[::-1]` and use the indices to reorder **both** `scores` and `names` together.
- **Starter:**
  ```python
  z = (scores - scores.mean(axis=0)) / scores.std(axis=0)
  order = scores.mean(axis=1).argsort()[::-1]
  sorted_names = names[order]
  ```

## 3. Linear algebra
- **Where:** [Linear algebra](../linear-algebra.md).
- **Think:** Live in `np.linalg.*` for det / inv / eigvals. Matrix multiplication is `@` or `np.matmul`. **Wrap `inv` in try/except `LinAlgError`** for singular matrices.
- **Starter:**
  ```python
  product = matrix_A @ matrix_B
  det_A   = np.linalg.det(matrix_A)
  try:
      inv_A = np.linalg.inv(matrix_A)
  except np.linalg.LinAlgError:
      inv_A = None
  ```

## 4. Advanced operations
- **Where:** [ndarray methods](../ndarray-methods.md) — broadcasting & unique, [boolean indexing](../boolean-indexing.md).
- **Think:**
  - **Broadcasting +5 to math column:** `scores[:, 0] += 5` adds a scalar to one column.
  - **Unique scores across all subjects:** `np.unique` flattens by default.
  - **Above average in all subjects:** same pattern as Q1.4 — compare to the per-subject mean (`axis=0`), then reduce with `.all(axis=1)`.

## Bonus — `student_analysis(name)`
- **Where:** [boolean indexing](../boolean-indexing.md) — `np.where`.
- **Think:** Three parts: get scores, compute rankings per subject, check top-3 by average.
  - Locate the student row: `idx = np.where(names == name)[0][0]`.
  - Ranking per subject = number of students who scored ≥ this student in that column. `rankdata`-style logic; a simple `(scores[:, j] >= s).sum()` works.
  - Top-3 by average: `argsort(avg)[-3:]` gives the three highest indices.

## Common pitfalls
- Mixing `axis=0` and `axis=1` — print `result.shape` after every reduction to confirm.
- Forgetting that boolean indexing **returns a copy**, not a view — assigning into it does nothing.
- Using `*` for matrix multiply (that's element-wise) — use `@` or `np.matmul`.

</details>
