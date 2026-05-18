# 1.2 Coding — Hints

For each task we point at the lesson section and give a **way to start** plus the first 1–2 lines of a starter pattern. We never give the full solution.

> Open [the assignment](./coding.md) in another tab.

## Task 1 — data types and variables
- **Where:** [Basic syntax & data types](../basic-syntax-data-types.md), [Data structures](../data-structures.md).
- **Think:** One assignment per type, one `print` to verify. Mix in a `print(type(x))` to confirm Python inferred the type you expected.
- **Starter:**
  ```python
  age = 30                     # int
  pi = 3.14                    # float
  fruits = ['apple', 'pear']   # list
  # ...continue for each type, then print()
  ```

## Task 2 — `count_and_return_vowels(text)`
- **Where:** [Conditions & iterations](../conditions-iterations.md), [Basic syntax & data types](../basic-syntax-data-types.md) — string methods.
- **Think:** You need two things — a **count** and a **list** of the matched characters. Iterate the text once. Compare each char against a vowels set (lowercase the char first if you want case-insensitive matching but still want to preserve the **original** case in the output list — read the expected output carefully).
- **Starter:**
  ```python
  vowels = set('aeiou')
  found = [c for c in text if c.lower() in vowels]
  return len(found), found
  ```

## Task 2 — `sum_of_even_numbers(limit)`
- **Where:** [Conditions & iterations](../conditions-iterations.md) — `while` loops.
- **Think:** The spec says **use a `while` loop** (not a comprehension). Decide whether `limit` is inclusive — the example `sum_of_even_numbers(10) == 30` (i.e. 2+4+6+8+10) tells you it is.
- **Starter:**
  ```python
  total, n = 0, 0
  while n <= limit:
      if n % 2 == 0:
          total += n
      n += 1
  return total
  ```

## Task 2 — `BankAccount` class
- **Where:** [Classes & objects](../classes-objects.md).
- **Think:** Four methods: `__init__`, `deposit`, `withdraw`, `get_balance`. State lives on `self.balance`. `withdraw` needs a guard — and the spec says to **print** "Insufficient funds" when the balance would go negative, not raise.
- **Starter:**
  ```python
  class BankAccount:
      def __init__(self, initial_balance):
          self.balance = initial_balance
      def deposit(self, amount):
          self.balance += amount
      # withdraw, get_balance — your turn
  ```

## Common pitfalls
- Forgetting `self` in method signatures.
- Returning `None` because you wrote `total += n` without a final `return`.
- Mutating a list while iterating it (build a new list instead).
