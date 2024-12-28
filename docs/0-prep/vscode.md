# Getting Started with VS Code for Python Development

Visual Studio Code (VS Code) is a lightweight but powerful source code editor that runs on your desktop. It comes with built-in support for JavaScript, TypeScript, and Node.js, and has a rich ecosystem of extensions for other languages including Python.

> **Note:** This guide assumes you have Anaconda installed on your system. If not, please refer to the [Anaconda installation instructions](./anaconda.md) before proceeding.

## Setting Up Python in VS Code

1. Install the Python extension:

   - Open VS Code
   - Click on the Extensions icon in the sidebar
   - Search for "Python"
   - Install the Microsoft Python extension

2. Configure Anaconda Integration:
   - If you have Anaconda installed, VS Code should automatically detect it
   - To verify your Python interpreter:
     - Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS)
     - Type "Python: Select Interpreter"
     - Look for your Anaconda environment (usually shows as "base")

## Verifying Anaconda Environment

To check if VS Code is using your Anaconda environment:

1. Open an existing Python file (`.py` extension) or create a new file, write `print("Hello, VS Code!")`, and save it as `hello.py`
2. Look at the bottom status bar - you should see your Python version and environment
3. Open a new terminal in VS Code (`Terminal > New Terminal`)
4. You should see `(base)` at the beginning of your terminal prompt

## Useful VS Code Features for Python

- IntelliSense: Autocomplete and syntax highlighting
- Linting: Code error checking
- Debugging: Built-in debugger
- Jupyter Notebooks: Direct support for `.ipynb` files
  > However, we will be using Jupyter Notebook in the Anaconda Navigator in this course
- Git Integration: Version control support

You can now start coding in Python with VS Code!
