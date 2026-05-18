# 1.4 Coding — Hints

For each task we point at the lesson section and give a **way to start** plus a starter pattern. We never give the full solution.

> Open [the assignment](./coding.md) in another tab.

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
