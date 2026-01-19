# Introduction to Jupyter Notebook

## What is Jupyter Notebook?

Jupyter Notebook is a **free** web-based tool that lets you create documents that combine code, text, images, and charts all in one place. It's the most popular tool for data science work!

**In simple terms:** Think of Jupyter Notebook as a digital lab notebook. Instead of writing code in separate files and results in different places, everything lives together - your code, explanations, results, and visualizations are all in one document.

**Why use Jupyter Notebook?**
- ✅ **Interactive** - Run code and see results immediately
- ✅ **Visual** - Create charts and graphs right in your notebook
- ✅ **Documentation** - Write explanations alongside your code
- ✅ **Shareable** - Easy to share with others
- ✅ **Industry standard** - Used by data scientists worldwide

![Jupyter Notebook Interface Placeholder - Shows a notebook with code and output]

> **Note:** This guide assumes you have Anaconda or `uv` installed. If not, please refer to the `Getting Started with Anaconda` or `Python for Data Science` guides first.

## Understanding Notebooks

### What is a Cell?

A notebook is made up of **cells** - think of them as building blocks. Each cell can contain either code or text.

### Cell Types

**1. Code Cells** (for writing Python code):
- Type your Python code here
- Press `Shift + Enter` to run the code
- Results appear directly below the cell
- You can run cells in any order (but usually top to bottom)

**2. Markdown Cells** (for writing explanations):
- Write text, explanations, and notes
- Supports formatting (bold, italics, headers, lists)
- Can include images, links, and even math equations
- Great for documenting what your code does

![Jupyter Cell Types Placeholder - Shows code cell vs markdown cell]

> **Tip:** You can mix code and markdown cells to create a story with your data analysis!

## Running Your Code

**How to execute (run) a cell:**

1. **Click the "Run" button** in the toolbar, OR
2. **Press `Shift + Enter`** (runs the cell and moves to the next one)
3. **Press `Ctrl + Enter`** (runs the cell but stays on the same cell)
4. **Press `Alt + Enter`** (runs the cell and creates a new cell below)

> **Tip:** The most common way is `Shift + Enter` - it runs your code and automatically moves to the next cell!

![Jupyter Run Button Placeholder - Shows the Run button in the toolbar]

## Keyboard Shortcuts (Hotkeys)

Jupyter has two modes, and different shortcuts work in each:

### Command Mode (Blue Border)
Press `Esc` to enter command mode. The cell border turns blue.

**Most Useful Shortcuts:**
- `a` - Insert a new cell **above** the current one
- `b` - Insert a new cell **below** the current one
- `d, d` - **Delete** the current cell (press 'd' twice)
- `z` - **Undo** cell deletion
- `m` - Change cell to **Markdown** (for text/explanations)
- `y` - Change cell to **Code** (for Python code)

### Edit Mode (Green Border)
Press `Enter` to enter edit mode. The cell border turns green.

**Most Useful Shortcuts:**
- `Shift + Enter` - Run cell and move to next
- `Ctrl + Enter` - Run cell (stay on same cell)
- `Alt + Enter` - Run cell and create new cell below

> **Tip:** Don't try to memorize all shortcuts at once! Start with `Shift + Enter` to run cells, and `a`/`b` to add cells. You'll learn the rest as you go.

![Jupyter Modes Placeholder - Shows command mode vs edit mode]

## Installation

### If You're Using Anaconda

Jupyter Notebook comes pre-installed with Anaconda! Just:
1. Open Anaconda Navigator
2. Click "Launch" under Jupyter Notebook

OR use the terminal:
```bash
# Make sure your environment is activated
conda activate dsai

# Launch Jupyter Notebook
jupyter notebook
```

### If You're Using `uv`

```bash
# Step 1: Activate your virtual environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Step 2: Install Jupyter
uv pip install jupyter notebook

# Step 3: Launch Jupyter Notebook
jupyter notebook
```

**What happens next:**
- Your web browser will open automatically
- You'll see the Jupyter file browser
- Navigate to your project folder
- Click "New" → "Python 3" to create a new notebook

![Jupyter File Browser Placeholder - Shows the Jupyter home page with file list]

## Useful Resources

- [Official Jupyter Documentation](https://jupyter.org/)
- [Jupyter Notebook Tutorial](https://www.dataquest.io/blog/jupyter-notebook-tutorial/)
- [Markdown Guide](https://www.markdownguide.org/)
- [Jupyter Notebook Shortcuts](https://towardsdatascience.com/jypyter-notebook-shortcuts-bf0101a98330)
