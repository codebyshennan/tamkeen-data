# Python for Data Science

This guide covers the installation and setup of essential Python libraries for data science. We'll use either Anaconda (recommended) or `uv` for package management.

## Prerequisites

* Python 3.10 installed
* Package manager: Anaconda or `uv` (see respective setup guides)

## Option 1: Using Anaconda (Recommended)

### Creating a New Environment

```bash
# Create a new environment with Python 3.10
conda create -n dsai python=3.10

# Activate the environment
conda activate dsai
```

### Installing Core Libraries

```bash
# Install scientific computing stack
conda install numpy pandas scipy

# Install visualization libraries
conda install matplotlib seaborn plotly

# Install machine learning libraries
conda install scikit-learn statsmodels

# Install data engineering libraries
conda install sqlalchemy requests pytest
conda install -c conda-forge great-expectations

# Install Jupyter
conda install jupyter notebook
```

### Verifying Installation

```python
# Run this in Python or Jupyter to verify installations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import statsmodels.api as sm

print(f"numpy version: {np.__version__}")
print(f"pandas version: {pd.__version__}")
print(f"matplotlib version: {plt.__version__}")
print(f"seaborn version: {sns.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")
print(f"statsmodels version: {sm.__version__}")
```

## Option 2: Using `uv`

### Creating a New Environment

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### Installing Core Libraries

```bash
# Install all required packages
uv pip install numpy pandas matplotlib seaborn scikit-learn statsmodels jupyter sqlalchemy requests pytest great-expectations
```

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
