# IDE Setup

**After this guide:** VS Code opens Python files, uses your **conda** or **uv** interpreter (status bar), and you can run or debug a small script from the editor or integrated terminal.

## What is an IDE?

An **IDE** (Integrated Development Environment) is a code editor with extra features like error checking, autocomplete, and debugging tools. Think of it as a word processor, but for writing code instead of documents.

## Visual Studio Code (VS Code)

**What is VS Code?** VS Code is a **free** code editor made by Microsoft. It's lightweight, fast, and has amazing features for Python development.

**Why use VS Code?**
- ✅ **Free** - Completely free to use
- ✅ **Lightweight** - Doesn't slow down your computer
- ✅ **Extensible** - Thousands of free extensions
- ✅ **Great for Python** - Excellent Python support
- ✅ **Popular** - Used by millions of developers
- ✅ **Cross-platform** - Works on Windows, macOS, and Linux

> **On screen:** VS Code main window, editor, sidebar, and terminal.

> **Note:** This guide assumes you have Anaconda or **uv** installed. If not, start with the [Anaconda guide](./anaconda.md) or your course's Python setup materials.

## Helpful video

Official Microsoft **beginner tour of VS Code** (~7 minutes): interface, extensions, terminal, and a **Python** segment (install the Python extension and run a file). Follow the written steps below for interpreter selection and your own OS.

<iframe width="560" height="315" src="https://www.youtube.com/embed/B-s71n0dHUk" title="Learn Visual Studio Code in 7min (Official Beginner Tutorial)" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Setting Up Python in VS Code

> **Time needed:** About 5 minutes

### Step 1: Install VS Code

If you don't have VS Code yet:

1. Visit [code.visualstudio.com](https://code.visualstudio.com/)
2. Download for your operating system
3. Run the installer and follow the prompts
4. Launch VS Code

> **On screen:** VS Code download or first-run screen for your OS.

### Step 2: Install the Python Extension

**Extensions add features to VS Code.** The Python extension is essential for Python development.

1. Open VS Code
2. Click the **Extensions icon** in the left sidebar (looks like four squares, or press **Ctrl+Shift+X** on Windows/Linux or **Cmd+Shift+X** on macOS)
3. In the search box, type: **"Python"**
4. Find **"Python"** by Microsoft (it should be the first result)
5. Click the **"Install"** button
6. Wait for installation to complete

> **On screen:** Extensions view with the Microsoft Python extension selected.

> **Tip:** VS Code will suggest installing the Python extension automatically when you open a **.py** file!

### Step 3: Configure Your Python Environment

**What is a Python interpreter?** It's the Python program that runs your code. VS Code needs to know which one to use.

**Automatic Detection:**
- If you have Anaconda: VS Code usually detects it automatically
- If you're using **uv**: VS Code will find your **.venv** folder

**Manual Selection (if needed):**

1. Press **Ctrl+Shift+P** (Windows/Linux) or **Cmd+Shift+P** (macOS)
2. Type: **"Python: Select Interpreter"**
3. Press Enter
4. You'll see a list of Python environments (examples):
   - Anaconda: **Python 3.12.x ('base': conda)** or **('dsai': conda)** (versions vary)
   - **uv**: **Python 3.12.x ('.venv': venv)** (versions vary)
5. Click on the one you want to use

> **On screen:** Command Palette open to **Python: Select Interpreter** with a list of environments.

> **Tip:** You can see which interpreter is active in the bottom-right corner of VS Code!

## Verifying Python Environment

To check if VS Code is using your Python environment:

1. Open or create a Python file named **hello.py**, add a simple **print** statement, and save. For example:

```python
print("Hello from your course environment!")
```
```
Hello from your course environment!
```


2. Look at the bottom status bar, you should see your Python version and environment
3. Open a new terminal: **Terminal → New Terminal**
4. You should see your environment name at the beginning of your terminal prompt:
   * Anaconda: **(base)** or **(dsai)** (or your conda env name)
   * **uv**: **(.venv)** or your virtual environment name

## Useful VS Code Features for Python

#### In-built features

* IntelliSense: Autocomplete and syntax highlighting
* Linting: Code error checking
* Debugging: Built-in debugger
* Git Integration: Version control support

#### Extensions

* Jupyter Notebooks: Direct support for **.ipynb** files

    > However, we will be using Jupyter Notebook in the Anaconda Navigator in this course
* [**autoDocstring**](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring), quickly generate docstrings for Python functions
* [**Error Lens**](https://marketplace.visualstudio.com/items?itemName=usernamehw.errorlens), surface diagnostics inline on the line
* [**Even Better TOML**](https://open-vsx.org/extension/tamasfe/even-better-toml), TOML editing for **pyproject.toml** and similar files
* [**Jupyter**](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter), open and run **.ipynb** notebooks in VS Code
* [**Pylance**](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance), Python language server (types, imports, IntelliSense); often installed automatically with the Python extension
* [**Python Indent**](https://marketplace.visualstudio.com/items?itemName=KevinRose.vsc-python-indent), indentation helpers for Python blocks
* [**Ruff**](https://github.com/astral-sh/ruff), fast linter and **isort**-style import sorting
* [**View Image for Python Debugging**](https://marketplace.visualstudio.com/items?itemName=elazarcoh.simply-view-image-for-python-debugging), useful when debugging outside notebooks

## Gotchas

- **Status bar ≠ terminal**: the interpreter shown in the bottom-right status bar controls Run/Debug and the editor's IntelliSense. The integrated terminal is a separate shell; it will use system Python until you run `conda activate dsai` (or the path activates automatically via `conda init`). Check both.
- **Opened from Spotlight/Finder on macOS**: VS Code launched outside a terminal doesn't source `.zshrc` or `.bash_profile`, so `conda` commands may not be found in the integrated terminal. Fix: open VS Code with `code .` from a terminal that already has the env active, or run `conda activate dsai` manually in the VS Code terminal.
- **"Python: Select Interpreter" is not sticky across workspaces**: each folder/workspace stores its own interpreter setting. If you open a different project folder, rerun **Python: Select Interpreter** for that folder.
- **Pylance not installed**: Pylance (type-checking, auto-import) is a separate extension from the Python extension. It's usually auto-installed alongside it, but if IntelliSense feels weak, check that Pylance is enabled in Extensions.
- **Jupyter extension kernel differs from browser Jupyter**: if you have both VS Code Jupyter and browser Jupyter open on the same file, each may use a different kernel. Stick to one UI per session; the kernel selector in the top-right of the notebook shows which Python is active.
- **Extension host crash**: VS Code occasionally crashes the extension host, especially on first install. Close and reopen VS Code; it usually self-heals.

## PyCharm (by JetBrains)

> **On screen:** PyCharm (optional), ML-aware completion or project view.
