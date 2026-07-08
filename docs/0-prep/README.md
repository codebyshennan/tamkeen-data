# Getting started: setup guides

**After this folder (essential path):** you meet every bullet under **You are ready when** below, a working env, imports, Jupyter, and ideally VS Code tied to the same interpreter.

Welcome to the Data Science and AI bootcamp. The pages in this folder get your **Python environment**, **libraries**, **editor**, and **notebooks** working so lesson time can focus on ideas instead of install errors. When something breaks, skim **[Gotchas](#gotchas)** before reinstalling from scratch.

**Start here:** if you have done nothing yet, do only this sequence today: (1) install [Anaconda](./anaconda.md) or [uv](./anaconda.md), (2) install the [Python data science stack](./python-ds-stack.md), (3) open [Jupyter](./jupyter-notebook.md) and run a one-cell "hello" notebook. Add [VS Code](./vscode.md) when you want a full editor for `.py` files. Everything else on this page is optional until a module tells you to install it.

> **On Windows?** Follow the [Windows setup guide](./windows.md), it consolidates all Windows-specific steps and gotchas into one place.

## Helpful video

Short overview of **what data science is** and how teams use data (about 8 minutes), context before setup, not a substitute for the lessons.

<iframe width="560" height="315" src="https://www.youtube.com/embed/RBSUwFGa6Fk" title="What is Data Science?" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## You are ready when

You can check all of the following:

- You have a **conda env** or **uv `.venv`** you know how to activate, and you use it every time you open a terminal.
- `python -c "import pandas, numpy, matplotlib"` runs with **no** `ModuleNotFoundError`.
- You can **start Jupyter** (from Anaconda Navigator or the command line) and run a cell that prints something.
- Optional but recommended: VS Code opens a `.py` file and uses the **same** interpreter as your course environment (status bar shows the env).

If any box fails, stay on that step until it passes, adding more tools rarely fixes a broken base install.

## Quick start (order matters)

1. **Python environment** (pick one):
   - [Anaconda](./anaconda.md) if you are new to Python or want a GUI.
   - **uv** only if you are already comfortable in the terminal, follow the same guide.
2. **Libraries**: [Python data science stack](./python-ds-stack.md)
3. **Editor**: [VS Code](./vscode.md) (recommended for scripts)
4. **Notebooks**: [Jupyter](./jupyter-notebook.md)

> **Note:** GitHub-style `- [ ]` task lists do not render as checkboxes on the course site, so use your own checklist or notes to track progress.

## Essential tools

### Must have

0. **[Windows setup guide](./windows.md)** *(Windows users only)*, start here before anything else

1. **[Anaconda or uv](./anaconda.md)**: isolates project packages
   - **Time:** about 15-20 minutes
   - Prefer Anaconda if you are new to programming.

2. **[Python data science libraries](./python-ds-stack.md)**: NumPy, pandas, plotting, sklearn, etc.
   - **Time:** about 10-15 minutes after the env exists

3. **[Jupyter Notebook](./jupyter-notebook.md)**: where most early exercises run
   - **Time:** usually zero extra install if you use Anaconda with Jupyter included

### Recommended

4. **[VS Code](./vscode.md)**: editing and debugging `.py` files
5. **[DBeaver](./dbeaver.md)**: connect to databases for SQL lessons (any SQL client you prefer is fine)

### Optional (install when a module needs them)

| Guide | Use case |
|-------|----------|
| [Google Colab](./google-colab.md) | Browser notebooks, no local install |
| [Tableau Public](./tableau.md) | BI / visualization modules |
| [Snowflake](./snowflake.md) | Cloud warehouse and advanced SQL |
| [Databricks](./databricks.md) | Spark / big data lessons |
| [Airflow](./airflow.md) | Workflow and scheduling labs |

## Suggested timeline

**Week 1, core setup:** conda or uv, libraries, confirm Jupyter, then VS Code.

**Later, as assigned:** DBeaver when SQL starts; Tableau, Snowflake, Databricks, or Airflow when the module syllabus says so.

## Gotchas

Cross-cutting pitfalls that show up before you open a specific tool guide. If something fails, check **activation** and **which interpreter/kernel** is running before reinstalling everything.

### Environment and "which Python?"

- **Wrong Python**: Many machines have several Pythons (system, Homebrew, Anaconda, an old installer). Always **activate** your course env (`conda activate …` or `source .venv/bin/activate`) *before* `pip`, `conda install`, or `uv pip install`, and check with `which python` / `python -V` when in doubt.
- **"Command not found"**: The tool is not on your `PATH`. Restart the terminal after install, open **Anaconda Prompt** on Windows if plain PowerShell does not see `conda`, or use the full path your guide mentions.
- **Permission errors**: On macOS/Linux, avoid `sudo pip` or `sudo` for envs inside your home folder; that mixes system Python with user envs. Fix folder ownership or reinstall as a normal user.
- **Breaking `base`**: In Anaconda, installing lots of packages into **`base`** can make conflicts harder to unwind. Prefer a **named env** (e.g. `dsai`) for course work and leave `base` minimal.
- **Mixing conda and pip**: In a conda env, prefer **`conda install`** for heavy scientific wheels when possible; use **`pip`** only when the package is not on conda-forge. Random `pip install` into the wrong env is a common source of "it worked yesterday."

### Jupyter and notebooks

- **Kernel ≠ terminal**: Jupyter uses a **kernel** tied to one Python. If `import pandas` fails in a notebook but works in a terminal, you installed packages into a **different** environment than the notebook's kernel. Fix: pick the right kernel (or install Jupyter *inside* the env you use for the course, as in the guides).
- **Cells out of order**: You can run cell 10 before cell 2; the notebook will still show old outputs. When results look impossible, use **Restart kernel and run all** after big changes.
- **Browser vs VS Code**: Opening the same `.ipynb` in the browser and in VS Code can use **different** kernels if each was configured separately. Stick to one workflow per project until you are comfortable switching.

### VS Code

- **Status bar vs integrated terminal**: The interpreter shown in the **status bar** applies to the editor and "Run Python File." The **terminal** might still be using system Python until you `conda activate` or open a new terminal from a folder where the right env is default.
- **Workspace trust / multiple roots**: In a monorepo or folder with several `.venv` directories, explicitly **Python: Select Interpreter** per workspace folder so the extension does not guess wrong.

### Installs and networks

- **Installs take a long time**: Normal for large scientific stacks; use a stable connection and avoid canceling mid-solve.
- **Corporate networks and SSL**: If `conda`/`pip` fails with certificate or proxy errors, you may need your IT proxy settings; the generic "could not fetch" message is not always a bad password.

### Optional tools (Colab, BI, cloud)

- **Google Colab**: Runtimes are **ephemeral**; installed packages disappear when the session ends unless you reinstall or use Colab's persistence patterns. Versions also differ from your local env.
- **Tableau / Snowflake / Databricks / Airflow**: Often tied to **accounts, trials, or org SSO**. Start signup early; "module week" is a bad time to discover a locked domain or pending IT approval.

## Getting help

1. **[Gotchas](#gotchas)** above, then troubleshooting sections in each guide
2. [How this course works](./pedagogy.md), what to bring to office hours
3. Your instructor's syllabus and announced channels

## How teaching works here

We use a **flipped** model: you study materials before live sessions and use class time to practice. Read [How this course works](./pedagogy.md) for a concrete week-by-week pattern and how to use office hours well.

## Tips

1. Finish core setup before racing ahead, broken envs waste more time than slow installs.
2. After each install, run the **verification** step in that guide.
3. Keep a short log of errors and fixes; it speeds up help.
4. Use one folder or repo for course work so paths stay predictable.

## Need more help?

- Each guide links to official documentation for the tool.
- Contributors: [Documentation guidelines](https://github.com/codebyshennan/tamkeen-data/blob/main/docs/meta/DOCUMENTATION_GUIDELINES.md) (the `docs/meta` folder is not published on the site).
- Grading and policies: your instructor and syllabus.

---

**Next step:** [Anaconda or uv](./anaconda.md), or [How this course works](./pedagogy.md) if your environment is already working.
