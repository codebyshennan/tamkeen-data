# **Assignment: Introduction to Python**

> **Submit your work on Skills Union →** <https://skillsu.com/member/assessment>

## **Objective**

Test your understanding of the majority of the concepts covered in the Python fundamentals lessons.

## **Instructions**

Complete the following tasks and submit your code in a Python script.

## **Task 1: Data Types and Variables**

Create a Python program that demonstrates the use of the following data types:

- Integers
- Floats
- Strings
- Boolean
- Lists
- Tuples
- Dictionaries

Use variables to store and print them.

### Examples

```python
x = 5  # integer
print(x)  # output: 5

y = 3.14  # float
print(y)  # output: 3.14
```

```
5
3.14
```

## **Task 2: Functions and Classes**

Complete the following Python codes at the "replace with your code" comments.

```python
def count_and_return_vowels(text):
    """
    Counts the number of vowels (a, e, i, o, u) in the given text (case-insensitive) and returns the count and list of vowels.
    """
    return 0, [] # replace with your code


print(count_and_return_vowels("Hello World")) # output: (3, ['e', 'o', 'o'])
print(count(count_and_return_vowels("Programming")) # output: (3, ['o', 'a', 'i'])
print(count_and_return_vowels("OpenAI")) # output: (2, ['O', 'e', 'A', 'I'])


def sum_of_even_numbers(limit):
    """
    Calculates the sum of even numbers up to a given limit using a while loop.
    """
    return 0 # replace with your code


print(sum_of_even_numbers(10)) # output: 30
print(sum_of_even_numbers(5)) # output: 6
print(sum_of_even_numbers(1)) # output: 0


class BankAccount:
    """
    Create a BankAccount class with:
    - Constructor that sets initial balance
    - deposit() method that adds money
    - withdraw() method that removes money if sufficient funds exist
    - get_balance() method that returns current balance
    """
    pass  # replace with your code


account = BankAccount(100)
print(account.get_balance())  # output: 100
account.deposit(50)
print(account.get_balance())  # output: 150
account.withdraw(30)
print(account.get_balance())  # output: 120
account.withdraw(200)  # Should print: "Insufficient funds"
print(account.get_balance())  # output: 120
```

## **Deliverable:**

Submit your code in a Python script (`.py` file). You can submit as a single file (`assignment.py`) or multiple files based on the tasks (`task1.py`, `task2.py`). Include comments to explain your code (optional).


## Hints

<details>
<summary>Show hints</summary>

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

</details>
