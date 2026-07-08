# Anaconda Setup

**After this guide:** you have Anaconda or **uv** installed, can create and activate an environment, and know how to install packages without mixing system Python and your course env.

## What is Anaconda?

Anaconda is a **free** Python distribution that comes with Python and hundreds of pre-installed data science packages. Think of it as a "starter kit" for data science - everything you need is already included!

**In simple terms:** Instead of installing Python and then installing dozens of packages one by one, Anaconda gives you everything in one package. It also includes a visual tool (Anaconda Navigator) that lets you manage everything with clicks instead of typing commands.

**Why use Anaconda?**

- ✅ **Beginner-friendly** - Visual interface (Anaconda Navigator) for managing packages
- ✅ **Pre-installed packages** - Popular data science tools ready to use
- ✅ **Easy environment management** - Create separate workspaces for different projects
- ✅ **Includes Jupyter Notebook** - Ready to use for data analysis
- ✅ **Well-supported** - Great documentation and community

> **On screen:** Anaconda Navigator home screen with **Launch** buttons.

## Helpful video

Short install walkthrough (download, run the installer, open Jupyter from Anaconda Navigator). The demo uses **Windows**; on macOS and Linux you still download the matching installer from [anaconda.com](https://www.anaconda.com/) and follow the same overall flow.

<iframe width="560" height="315" src="https://www.youtube.com/embed/wXyaCiL_cHg" title="How to Install Anaconda and Jupyter Notebook on Windows" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## What is uv?

**uv** is a modern, super-fast Python package manager. It's like a turbocharged version of **pip**, it installs packages 10-100 times faster!

**When to use uv:**

- You're comfortable with command-line tools
- You want faster package installation
- You prefer lightweight tools
- You're working on multiple projects

## Which Should I Choose?

**Choose Anaconda if:**

- You're a beginner
- You prefer visual tools (GUI)
- You want everything pre-installed
- You want the easiest setup experience

**Choose uv if:**

- You're comfortable with command-line
- You want faster package installation
- You prefer lightweight tools
- You're an experienced developer

> **Tip for beginners:** Start with Anaconda! You can always learn **uv** later. This guide covers both options.

## Option 1: Anaconda

### Why Anaconda?

- Simplifies package management and deployment
- Includes 1,500+ open source packages
- Includes the most popular data science tools like Jupyter Notebook
- Suitable for both beginners and advanced users
- Provides Anaconda Navigator GUI for easy management

### Download and Installation

> **Time needed:** About 15-20 minutes (download time varies)

**Step 1: Download Anaconda**

1. Visit the [Anaconda Download Page](https://www.anaconda.com/download)
2. You'll see options for different operating systems
3. Click **"Download"** for your operating system (choose 64-Bit)
4. The download will start - this file is large (500MB+), so it may take a while

> **On screen:** Anaconda download page with OS choices.

**Step 2: Install Anaconda**

<details>
<summary><b>Windows Installation</b></summary>

1. Find the downloaded **.exe** installer (usually in your Downloads folder)
2. Double-click to run the installer
3. If Windows asks for permission, click **"Yes"**
4. Follow the installation wizard:

   **Important Installation Options:**
   - **Install for:** Choose **"Just Me"** (recommended for most users)
   - **Destination folder:** Leave as default (usually **C:\\Users\\YourName\\anaconda3**)
   - **Advanced Options:**
     - ✅ **Check:** "Add Anaconda to my PATH environment variable"
     - ✅ **Check:** "Register Anaconda as my default Python"

5. Click **"Install"** and wait (this takes 5-10 minutes)
6. Click **"Next"** and then **"Finish"** when done

> **On screen:** Windows installer, **Just Me**, PATH, and finish screens.

> **Note:** If you see a warning about PATH, that's okay - you checked the box to add it automatically.

**Verify Installation:**

1. Open **Anaconda Prompt** from the Start menu (search "Anaconda Prompt")
2. Type: `conda --version` and press Enter
3. You should see something like `conda 24.x` or `25.x`

If you see a version number, Anaconda is installed correctly.

> **Troubleshooting:** If `conda` is not recognized, restart your computer and open Anaconda Prompt (not PowerShell or cmd), it sets PATH automatically.

</details>

<details>
<summary><b>macOS Installation</b></summary>

1. Find the downloaded **.pkg** file (usually in your Downloads folder)
2. Double-click to open the installer
3. Follow the installation prompts:
   - Click **"Continue"** through the introduction
   - Read (or scroll through) the license and click **"Agree"**
   - Choose **"Install for me only"** (recommended)
   - Click **"Install"** and enter your password when prompted
4. Wait for installation to complete (5-10 minutes)
5. Click **"Close"** when finished

> **On screen:** macOS installer, license, destination, password prompt.

**Verify Installation:**

1. Open **Terminal** (Applications → Utilities → Terminal)
2. Type: **conda --version**
3. Press Enter
4. You should see something like **conda 24.x** or **25.x** (version numbers vary by installer date)

If you see a version number, congratulations, Anaconda is installed.

> **Troubleshooting:** If you see "command not found", try restarting Terminal or your computer.

</details>

### Package Installation

**What is an environment?** Think of it as a separate workspace for your project. Each environment has its own set of packages, so you can have different versions for different projects without conflicts.

**Step 1: Open Terminal/Anaconda Prompt**

- **Windows:** Open "Anaconda Prompt" from the Start menu (search for "Anaconda Prompt")
- **macOS:** Open Terminal (Applications → Utilities → Terminal)

**Step 2: Create Your First Environment**

```bash
# Create a new environment named "dsai" with Python 3.12 (widely supported by scientific packages in 2025-2026)
# Use 3.11 or 3.13 if your instructor or workplace standardizes on them
conda create -n dsai python=3.12
```

When prompted, type `y` and press Enter to proceed.

**Step 3: Activate Your Environment**

```bash
# Activate the environment (you'll see "(dsai)" appear in your prompt)
conda activate dsai
```

> **What just happened?** You created a fresh workspace and activated it. Notice your prompt now shows **(dsai)**, that means you're working in that environment!

**Step 4: Install Essential Packages**

```bash
# Install core data science packages
# This will take several minutes - grab a coffee!
conda install numpy pandas matplotlib seaborn scikit-learn statsmodels jupyter
```

When prompted, type `y` and press Enter for each installation.

> **Tip:** You can install packages one at a time, but installing them together is faster and helps avoid conflicts.

For **Apache Airflow**, use a **separate** environment or a **uv** virtualenv: Airflow 3.x is installed with official [constraint files](https://airflow.apache.org/docs/apache-airflow/stable/installation/installing-from-pypi.html), and the version on `conda-forge` may not match the course. Follow the [Airflow setup guide](./airflow.md) when you reach that module.

> **On screen:** Terminal running **conda create** / **conda activate** / **conda install**.

### Environment Management

```bash
# List all environments
conda env list

# Export environment
conda env export > environment.yml

# Create environment from file
conda env create -f environment.yml

# Remove environment
conda env remove -n environment_name
```

## Launching Tools

### Anaconda Navigator (Anaconda only)

Anaconda Navigator is a visual tool that lets you launch applications and manage packages with clicks instead of commands.

**How to Open:**

- **Windows:** Start menu → Search "Anaconda Navigator" → Click it
- **macOS:** Launchpad → Search "Anaconda Navigator" → Click it

> **On screen:** Anaconda Navigator list of apps (Jupyter, VS Code, etc.).

**What you'll see:**

- A list of applications you can launch (Jupyter Notebook, VS Code, etc.)
- Package management tools
- Environment management

### Jupyter Notebook

**Method 1: Using Anaconda Navigator (Easiest for Beginners)**

1. Open Anaconda Navigator
2. Find "Jupyter Notebook" in the list
3. Click the **"Launch"** button below it
4. Your web browser will open automatically with Jupyter Notebook

> **On screen:** Jupyter Notebook row with **Launch** highlighted.

**Method 2: Using Terminal/Command Prompt**

1. Open Terminal (macOS) or Anaconda Prompt (Windows)
2. Make sure your environment is activated (you should see **(dsai)** in the prompt)
3. Type: **jupyter notebook** and press Enter
4. Your default browser will open with Jupyter Notebook

> **Tip:** If you see a URL like **<http://localhost:8888>**, that's normal, Jupyter is running on your computer!

### VS Code

**Using Anaconda Navigator:**

1. Open Anaconda Navigator
2. Find "VS Code" in the list
3. Click the **"Launch"** button
4. VS Code will open

> **Note:** If VS Code isn't listed, you may need to install it separately. See the VS Code setup guide for details.

## Gotchas

### Anaconda

- **"Add to PATH" is no longer recommended**: recent Anaconda installers default to *not* adding conda to PATH. Instead, use **Anaconda Prompt** on Windows or run `conda init zsh` / `conda init bash` once after install so your terminal shell can find `conda`. Restart the terminal afterwards.
- **`conda activate` says "no such command"**: this means `conda init` was never run for your shell. Fix: `conda init <shell>` (e.g. `conda init zsh`), then restart the terminal.
- **Solving environment takes forever**: the classic conda solver is slow on large environments. Enable the faster libmamba solver: `conda config --set solver libmamba` (requires `conda-libmamba-solver`, included in Anaconda 23.x+).
- **Anaconda Navigator launches base, not your course env**: Navigator apps launch in the `base` env by default. Use the **Environment** dropdown in Navigator to switch to `dsai` before launching Jupyter.
- **Don't install course packages into `base`**: conflicts in `base` are hard to untangle. Keep `base` minimal; do all course work in a named env (`dsai`).

### uv

- **"uv: command not found" after install**: the install script adds uv to `~/.local/bin` (Linux/macOS) or `%USERPROFILE%\.local\bin` (Windows) and updates your shell config, but the current terminal session doesn't see it yet. Close and reopen the terminal, or run `source ~/.zshrc` / `source ~/.bashrc`.
- **`uv venv` creates `.venv` in the current folder**: if you run it in the wrong directory, a stray `.venv` appears there. Always `cd` into your project folder first.
- **Forgetting to activate before installing**: `uv pip install` without activating `.venv` installs into the global Python. Check that `(.venv)` appears in your prompt before running any install commands.



1. **PATH Issues**:
   - Windows: Restart your computer
   - macOS: Run **source ~/.bash_profile** (or use **~/.zshrc** on newer macOS) or restart Terminal
2. **Package Conflicts**:

    ```bash
    # Remove conflicting package
    conda remove package_name

    # Reinstall with specific version
    conda install package_name=version
    ```

3. **Environment Activation Fails**:
   - Windows: Run as Administrator
   - macOS: Check **conda init** was run for your shell
4. **Jupyter Notebook Kernel Missing**:

    ```bash
    python -m ipykernel install --user --name=dsai
    ```

## Option 2: uv

### Why uv?

[**uv**](https://github.com/astral-sh/uv) is a modern Python package manager created by Astral (the team behind Ruff). It's designed to be **10-100 times faster** than traditional **pip**.

**Key Benefits:**

- **Super fast**: Installs packages much faster than pip
- **Better dependency resolution**: Handles package conflicts automatically
- **Lightweight**: Smaller footprint than Anaconda
- **All-in-one tool**: Replaces multiple tools (pip, venv, pip-tools, etc.)
- **Compatible**: Works with **requirements.txt** and **pyproject.toml**

**Who should use uv?**

- Developers comfortable with command-line tools
- People who want faster package installation
- Those who prefer lightweight tools
- Experienced Python developers

> **On screen:** Optional, timing **uv** vs **pip** for the same installs (for slides or motivation).

### Installation

> **Time needed:** Less than 2 minutes

**Step 1: Install uv**

Open your terminal and run the command for your operating system:

**For macOS/Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**For Windows (PowerShell):**

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**Step 2: Verify Installation**

Close and reopen your terminal, then type:

```bash
uv --version
```

You should see a version number (for example **uv 0.6.x** or newer, check [astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) if you need the latest). If you do, you're all set.

> **Troubleshooting:** If the command isn't found, you may need to restart your terminal or add **uv** to your PATH. Check the [uv documentation](https://github.com/astral-sh/uv) for platform-specific instructions.

### Usage

**Step 1: Create a New Environment**

Navigate to your project folder, then:

```bash
# Create a new virtual environment
# This creates a folder called ".venv" in your current directory
uv venv
```

**Step 2: Activate the Environment**

```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

You should see **(.venv)** in your terminal prompt, that means the environment is active!

**Step 3: Install Packages**

```bash
# Install all essential data science packages
# This is much faster than pip!
uv pip install numpy pandas matplotlib seaborn scikit-learn statsmodels jupyter
```

Install **Apache Airflow** in its own folder and virtual environment using the pinned, constraint-based steps in the [Airflow guide](./airflow.md), do not add it to your main `dsai` environment unless your instructor asks you to.

> **Tip:** You can install packages one at a time, or all together like above. **uv** will handle many conflicts automatically.

> **On screen:** Terminal showing **uv pip install** resolving packages.

### uv issues

1. **Installation Fails**:
   - Check Python version compatibility
   - Ensure build tools are installed
2. **Package Conflicts**:

    ```bash
    # Force reinstall of one package
    uv pip install --reinstall package_name
    ```

3. **Environment Issues**:
   - Delete the **.venv** folder and recreate
   - Check **PATH** settings

### General Tips

1. **Slow Package Installation**:
   - Use faster package mirrors
   - For Anaconda: **conda config --add channels conda-forge**
   - For uv: **uv pip install --cache-dir=.cache** (example cache location)
2. **Memory Issues**:
   - Close unnecessary applications
   - For large packages, try installing one at a time
3. **Version Conflicts**:
   - Create a new environment for different projects
   - Use **conda list** or **pip list** to check installed versions
4. **Jupyter Integration**:
   - Ensure kernels are properly registered
   - Check kernel paths match environment paths
