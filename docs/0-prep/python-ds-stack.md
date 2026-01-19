# Python for Data Science

## What is the Python Data Science Stack?

The "Python Data Science Stack" refers to a collection of essential libraries (pre-written code) that data scientists use every day. Instead of writing everything from scratch, these libraries provide powerful tools for:

- **Working with data** - Reading, cleaning, and analyzing datasets
- **Creating visualizations** - Making charts and graphs
- **Machine learning** - Building predictive models
- **Statistical analysis** - Performing statistical tests and calculations

**In simple terms:** Think of these libraries as specialized toolboxes. Each one has tools for specific tasks, and together they give you everything you need for data science work.

**Why install these?**
- ✅ **Essential tools** - You'll use these in almost every data science project
- ✅ **Industry standard** - These are the libraries used by professionals worldwide
- ✅ **Well-documented** - Great learning resources available
- ✅ **Active community** - Lots of help available online

![Python Data Science Stack Placeholder - Shows the main libraries and their purposes]

## Prerequisites

Before you begin, make sure you have:

* **Python 3.10 or newer** installed
* **A package manager** - Either Anaconda or `uv` (see the Anaconda setup guide)

> **Don't have Python yet?** If you install Anaconda, Python comes with it! If you're using `uv`, you'll need Python installed separately.

## Option 1: Using Anaconda (Recommended for Beginners)

> **Time needed:** About 10-15 minutes (depending on internet speed)

### Step 1: Create a New Environment

**What is an environment?** It's like a separate workspace that keeps your packages organized. This prevents conflicts between different projects.

```bash
# Create a new environment named "dsai" with Python 3.10
# Replace "dsai" with any name you prefer (e.g., "data-science", "my-project")
conda create -n dsai python=3.10
```

When prompted, type `y` and press Enter. This will take a minute or two.

### Step 2: Activate Your Environment

```bash
# Activate the environment
# You'll see "(dsai)" appear in your terminal prompt
conda activate dsai
```

> **Tip:** You need to activate your environment every time you open a new terminal window. The `(dsai)` in your prompt reminds you which environment you're using!

### Step 3: Install Core Libraries

**What are we installing?** Here's what each library does:

- **NumPy** - Working with arrays and mathematical operations
- **Pandas** - Reading and analyzing data (like Excel for Python)
- **Matplotlib & Seaborn** - Creating charts and visualizations
- **Scikit-learn** - Machine learning tools
- **Statsmodels** - Statistical analysis
- **Jupyter** - Interactive notebooks for data analysis

```bash
# Install scientific computing libraries
# These are the foundation of data science
conda install numpy pandas scipy

# Install visualization libraries
# For creating charts and graphs
conda install matplotlib seaborn plotly

# Install machine learning libraries
# For building predictive models
conda install scikit-learn statsmodels

# Install data engineering libraries
# For working with databases and APIs
conda install sqlalchemy requests pytest

# Install additional tools from conda-forge
conda install -c conda-forge great-expectations

# Install Jupyter Notebook
# For interactive data analysis
conda install jupyter notebook
```

> **Note:** Each `conda install` command will ask for confirmation. Type `y` and press Enter each time. The installation may take 5-10 minutes total.

![Package Installation Progress Placeholder - Shows packages being installed]

### Step 4: Verify Installation

**How to check if everything installed correctly:**

1. Make sure your environment is activated (you should see `(dsai)` in your terminal)
2. Start Python by typing: `python` and pressing Enter
3. Copy and paste this code:

```python
# Test if all libraries can be imported
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import statsmodels.api as sm

# Print version numbers
print("✅ All libraries installed successfully!")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Matplotlib version: {plt.__version__}")
print(f"Seaborn version: {sns.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Statsmodels version: {sm.__version__}")
```

4. You should see version numbers for each library
5. Type `exit()` to leave Python

> **Troubleshooting:** If you see an error like "ModuleNotFoundError", that library didn't install correctly. Try installing it again with `conda install library_name`.

![Python Verification Output Placeholder - Shows successful library imports]

## Option 2: Using `uv`

> **Time needed:** About 5-10 minutes (much faster than Anaconda!)

### Step 1: Create and Activate Environment

Navigate to your project folder, then:

```bash
# Create a new virtual environment
# This creates a ".venv" folder in your current directory
uv venv

# Activate the environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

You should see `(.venv)` appear in your terminal prompt - that means it's active!

### Step 2: Install All Libraries

**One command installs everything!** `uv` is much faster than conda:

```bash
# Install all essential data science libraries at once
# This will take a few minutes, but much faster than conda!
uv pip install numpy pandas matplotlib seaborn scikit-learn statsmodels jupyter sqlalchemy requests pytest great-expectations
```

> **Tip:** `uv` handles dependency conflicts automatically, so you don't need to worry about package versions!

![uv Fast Installation Placeholder - Shows quick package installation]

## Common Issues & Troubleshooting

### Installation Problems

1.  **Package Conflicts**:

    ```bash
    # Anaconda solution
    conda remove package_name
    conda install package_name=specific_version

    # uv solution
    uv pip uninstall package_name
    uv pip install package_name==specific_version
    ```
2. **Memory Issues During Installation**:
   * Close unnecessary applications
   * Install packages one at a time
   * Use smaller environment configurations
3. **Import Errors**:
   * Verify environment activation
   * Check package installation with `conda list` or `pip list`
   * Reinstall problematic package
   * For great\_expectations: ensure compatible Python version
4. **SQLAlchemy Connection Issues**:
   * Verify database URL format
   * Check database credentials
   * Test database connectivity
   * Install required database drivers

### Library-Specific Issues

1.  **Matplotlib Backend Problems**:

    ```python
    # Add to your notebook or script
    import matplotlib
    matplotlib.use('Agg')  # for no-GUI backend
    # or
    %matplotlib inline    # in Jupyter
    ```
2. **Numpy Performance Issues**:
   * Check if using optimized BLAS/LAPACK
   * Install `numpy` with MKL support via `conda`
3. **Pandas Memory Errors**:
   * Use appropriate data types (e.g., categories for strings)
   * Read large files in chunks
   * Use memory-efficient methods like `read_csv(..., usecols=[...])`

## Best Practices

### Environment Management

1.  **Project Organization**:

    ```
    project/
    ├── .gitignore
    ├── README.md
    ├── data/
    ├── notebooks/
    ├── src/
    └── requirements.txt
    ```
2.  **Dependencies Documentation**:

    ```bash
    # Anaconda
    conda env export > environment.yml

    # uv
    uv pip freeze > requirements.txt
    ```
3. **Version Control**:
   * Add virtual environment directories to .gitignore
   * Track requirements.txt or environment.yml
   * Document Python version used

### Performance Optimization

1. **Numpy**:
   * Use vectorized operations
   * Avoid Python loops when possible
   * Pre-allocate arrays
2. **Pandas**:
   * Use appropriate data types
   * Chain operations efficiently
   * Use vectorized operations
3. **Matplotlib/Seaborn**:
   * Close figures when not needed
   * Use style sheets for consistent plotting
   * Save plots in vector formats (SVG/PDF) for quality

## Additional Resources

1. **Documentation**:
   * [NumPy Documentation](https://numpy.org/doc/stable/)
   * [Pandas Documentation](https://pandas.pydata.org/docs/)
   * [Matplotlib Documentation](https://matplotlib.org/stable/)
   * [Seaborn Documentation](https://seaborn.pydata.org/)
   * [Scikit-learn Documentation](https://scikit-learn.org/stable/)
   * [StatsModels Documentation](https://www.statsmodels.org/stable/index.html)
2. **Tutorials**:
   * [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
   * [Real Python Tutorials](https://realpython.com/tutorials/data-science/)
   * [Scipy Lectures](https://scipy-lectures.org/)
3. **Community Support**:
   * Stack Overflow tags: \[numpy], \[pandas], \[matplotlib], \[scikit-learn]
   * GitHub Issues for respective libraries
   * Reddit: [`r/datascience`](https://www.reddit.com/search/?q=datascience), [`r/learnpython`](https://www.reddit.com/search/?q=learnpython)
