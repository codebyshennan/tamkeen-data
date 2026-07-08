# Getting Started with Databricks Community Edition

**After this guide:** you can log into Community Edition, create or open a **notebook**, attach **compute**, and run a Python or SQL cell successfully.

## What is Databricks?

Databricks is a cloud-based platform that lets you work with data and build machine learning models using Python, SQL, R, or Scala. The Community Edition is **completely free** and perfect for learning!

**In simple terms:** Think of Databricks as Google Colab's big brother - it's more powerful, designed for larger datasets, and includes tools for working with "big data" using Apache Spark.

**Key Benefits:**
- ✅ No installation needed - works in your web browser
- ✅ Free to use (with some limitations)
- ✅ Pre-installed with popular data science libraries
- ✅ Can handle much larger datasets than your local computer
- ✅ Great for learning Spark and distributed computing

> **On screen:** Databricks workspace with notebooks.

## Helpful video

Very short **what is Databricks?** overview (under 4 minutes): lakehouse idea at a high level. Use it alongside the sign-up steps below, this is not a product tutorial.

<iframe width="560" height="315" src="https://www.youtube.com/embed/GGqQqjLrJYI" title="what is databricks | introduction" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## System Requirements

- Modern web browser (Chrome, Firefox, Safari, or Edge)
- Stable internet connection
- No software installation required (cloud-based)
- GitHub account (for authentication)

## Key Features

- Interactive notebooks (supporting Python, SQL, R, and Scala)
- Small single-node cluster for computation
- Built-in libraries and tools
- Sample datasets and notebooks
- Integration with popular ML frameworks

## Account Setup

> **Time needed:** About 5 minutes

**Step 1: Visit the Sign-Up Page**
1. Go to [Databricks Community Edition](https://community.cloud.databricks.com/login.html)
2. Click the **"Get Started With Community Edition"** button

> **On screen:** Community Edition sign-up / login.

**Step 2: Choose Your Sign-Up Method**
- **Option A: GitHub** (Recommended - faster and easier)
  - Click "Sign up with GitHub"
  - Authorize Databricks to access your GitHub account
- **Option B: Email**
  - Enter your email address
  - Create a password
  - Complete the registration form

**Step 3: Complete Registration**
1. Fill out the registration form with your information
2. Verify your email address (check your inbox!)
3. Accept the terms of service
4. You're ready to go.

> **Tip:** If you don't have a GitHub account, you can create one for free at github.com - it's useful for many data science tools!

## Initial Configuration

### Workspace Setup

**Step 1: Create Your First Notebook**

A notebook is where you'll write and run your code - similar to Jupyter Notebooks!

1. Click **"Create"** in the left sidebar (or the "+" button)
2. Choose **"Notebook"** from the dropdown menu
3. Select **Python** as the default language (you can change this later)
4. Give your notebook a descriptive name (e.g., "My First Data Analysis")
5. Click **"Create"**

> **On screen:** Creating a new notebook.

**Step 2: Create a Cluster**

A cluster is like a remote computer that runs your code. You need one to execute your notebooks!

1. Click **"Compute"** in the left sidebar (looks like a computer/server icon)
2. Click the **"Create Cluster"** button
3. For Community Edition, the default settings are perfect - don't change anything!
4. Give your cluster a name (e.g., "My Learning Cluster")
5. Click **"Create Cluster"**
6. Wait 2-3 minutes for the cluster to start (you'll see a spinning icon)

> **On screen:** Cluster / compute creation (size, runtime).

> **Important:** Your cluster will automatically stop after 2 hours of inactivity to save resources. Just click "Start" when you need it again!

### Importing Libraries

Common data science libraries are pre-installed. Import them in your notebook:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
```

## Best Practices

### Workspace Organization

1. **Project Structure**:
   ```text
   Workspace/
   ├── Shared/
   │   ├── Projects/
   │   └── Libraries/
   └── Users/
       └── your.email@domain.com/
           ├── Project1/
           └── Project2/
   ```

2. **Notebook Management**:
   - Use meaningful names
   - Add descriptions
   - Regular checkpoints
   - Version control integration

### Performance Optimization

1. **Cluster Usage**:
   - Start cluster before use
   - Stop when not in use
   - Monitor cluster health

2. **Code Efficiency**:
   - Use Spark DataFrame operations
   - Minimize data movement
   - Cache frequently used data

## Gotchas

- **Cluster auto-terminates after 2 hours**: Community Edition clusters stop automatically after 2 hours of inactivity. When you return, click **Start** on the cluster before running any notebook cells. You will not lose your notebooks, but in-memory variables and installed packages are gone.
- **Notebooks must be attached to a running cluster**: the cluster status indicator at the top of the notebook must show "Attached" (not "Detached" or "Starting"). If it's detached, click the cluster name and select your running cluster.
- **`%pip install` is session-scoped**: packages installed with `%pip install` are available for the current cluster session only. When the cluster restarts, you must re-run those cells. Put install cells near the top of your notebook so they run as part of the setup.
- **Community Edition has no persistent /tmp storage**: files written to the local filesystem outside DBFS disappear when the cluster restarts. Use `dbutils.fs.put(...)` or write to `/dbfs/...` paths for anything you want to keep between sessions.
- **PySpark ≠ pandas**: `spark.read.csv(...)` returns a Spark DataFrame, not a pandas DataFrame. `.show()` replaces `.head()`, and many pandas methods don't exist on Spark DataFrames. Use `.toPandas()` to convert when needed.
- **GitHub auth may ask for re-authorization**: if you signed up via GitHub and Databricks loses the token, it will prompt for re-authorization. Check your GitHub OAuth apps if this keeps happening.
- **Community Edition ≠ production Databricks**: the free tier is a single-node setup; there is no true distributed computing. Code written here will run on actual clusters, but do not expect the same concurrency or scale.



### Connection Problems

1. **Cluster Not Starting**:
   - Check quota limits
   - Verify cluster configuration
   - Restart browser
   - Clear browser cache

2. **Notebook Not Connecting**:
   - Ensure cluster is running
   - Detach and reattach notebook
   - Restart cluster

### Runtime Issues

1. **Out of Memory**:
   - Reduce data size
   - Optimize queries
   - Clear notebook state

2. **Slow Performance**:
   - Check network connection
   - Optimize code
   - Monitor cluster metrics

## Tips for Success

1. **Learning Resources**:
   - Complete quickstart tutorials
   - Review sample notebooks
   - Join community forums

2. **Development Workflow**:
   - Use notebook cells effectively
   - Document your code
   - Regular commits to version control

3. **Data Management**:
   - Upload data through UI
   - Use DBFS for storage
   - Implement proper cleanup

## Additional Resources

1. **Documentation**:
   - [Databricks Documentation](https://docs.databricks.com/)
   - [Databricks documentation](https://docs.databricks.com/) (search for *Community Edition* topics)
   - [Spark Documentation](https://spark.apache.org/docs/latest/)

2. **Learning Materials**:
   - [Databricks Academy](https://academy.databricks.com/)
   - [Example Notebooks](https://docs.databricks.com/notebooks/notebooks-use.html)
   - [Community Forums](https://community.databricks.com/)

3. **Support Channels**:
   - Community Forums
   - Stack Overflow
   - GitHub Issues
