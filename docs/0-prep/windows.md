# Windows Setup Guide

**After this guide:** you have a working terminal, Python environment (Anaconda or uv), data science libraries, Jupyter, and VS Code on Windows, ready for the first lesson.

This guide consolidates Windows-specific steps and quirks into one place. Skim the cross-platform guides ([Anaconda](./anaconda.md), [Python stack](./python-ds-stack.md), [VS Code](./vscode.md)) for background, then follow this page end-to-end.

## Helpful video

Full Windows setup walkthrough, Anaconda, Jupyter, and VS Code on a fresh Windows machine (about 10 minutes):

<iframe width="560" height="315" src="https://www.youtube.com/embed/wXyaCiL_cHg" title="How to Install Anaconda and Jupyter Notebook on Windows" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Step 0: Pick a terminal

Windows has three common shells. You need to know which one you're using.

| Shell | How to open | Use when |
|-------|-------------|----------|
| **Anaconda Prompt** | Start → search "Anaconda Prompt" | Running `conda` commands, most reliable |
| **PowerShell** | Start → search "PowerShell" | General scripting; `conda` works after `conda init powershell` |
| **Command Prompt (cmd)** | Start → search "cmd" | Fallback only; avoid for this course |

> **Recommendation for beginners:** use **Anaconda Prompt** for all `conda` and `python` commands until you are confident in your setup.

**Optional, Windows Terminal:** Microsoft's modern tabbed terminal. Install it from the [Microsoft Store](https://aka.ms/terminal); it lets you run Anaconda Prompt, PowerShell, and cmd in separate tabs.

## Step 1: Install Anaconda (recommended for beginners)

> **Time needed:** 15-20 minutes (download time varies)

**Download:**

1. Go to [anaconda.com/download](https://www.anaconda.com/download)
2. Click the **Download** button, it auto-detects Windows and picks the 64-bit installer
3. The file is 500 MB+; wait for the download to finish

**Install:**

1. Double-click the downloaded `.exe` file
2. If Windows shows a User Account Control prompt, click **Yes**
3. Work through the wizard:

   | Option | What to choose |
   |--------|----------------|
   | **Installation type** | "Just Me" (recommended) |
   | **Destination folder** | Leave default (`C:\Users\YourName\anaconda3`) |
   | **Add Anaconda to PATH** | **Leave unchecked** (the installer now warns against it; use Anaconda Prompt instead) |
   | **Register as default Python** | Check this if you want `python` in PowerShell to point to Anaconda |

4. Click **Install** and wait 5-10 minutes
5. Click **Finish** when done

**Verify:**

Open **Anaconda Prompt** from the Start menu and type:

```bash
conda --version
```

You should see something like `conda 24.x` or `25.x`. If not, restart your computer and try again.

## Step 2: Create your course environment

All commands below run in **Anaconda Prompt** unless noted.

```bash
# Create a new environment named "dsai" with Python 3.12
conda create -n dsai python=3.12
```

Type `y` and press Enter when prompted.

```bash
# Activate the environment: you should see "(dsai)" in your prompt
conda activate dsai
```

> **Every new terminal session:** you must run `conda activate dsai` again. The prompt always shows the active environment in parentheses.

## Step 3: Install data science libraries

```bash
# Core libraries
conda install numpy pandas scipy matplotlib seaborn scikit-learn statsmodels

# Jupyter
conda install jupyter notebook
```

Type `y` when asked. Total install time is 5-10 minutes.

**Verify:**

```bash
python -c "import numpy, pandas, matplotlib, sklearn; print('All good!')"
```

You should see `All good!`. If you see `ModuleNotFoundError`, run the install step again for the missing package.

## Step 4: Launch Jupyter Notebook

**Option A, Anaconda Navigator (easiest):**

1. Open Anaconda Navigator from the Start menu
2. In the **Environment** dropdown (left panel), switch from `base` to `dsai`
3. Click **Launch** under Jupyter Notebook

**Option B, Anaconda Prompt:**

```bash
conda activate dsai
jupyter notebook
```

Your browser opens automatically. Navigate to your project folder, then click **New → Python 3** to create a notebook.

## Step 5: Install VS Code

1. Download from [code.visualstudio.com](https://code.visualstudio.com/), choose the **System Installer** (not User Installer) for best PATH integration
2. Run the installer; on the **Select Additional Tasks** screen, check:
   - ✅ "Add to PATH (requires shell restart)"
   - ✅ "Add 'Open with Code' action to Windows Explorer"
3. Launch VS Code after install

**Install the Python extension:**

1. Press `Ctrl+Shift+X` to open Extensions
2. Search for **Python** (Microsoft)
3. Click **Install**

**Point VS Code at your course environment:**

1. Press `Ctrl+Shift+P`, type **Python: Select Interpreter**, press Enter
2. Select **Python 3.12.x ('dsai': conda)**

You should now see `dsai` in the bottom-right status bar.

## Alternative: uv (for experienced users)

If you prefer a lightweight setup without Anaconda:

**Install uv (PowerShell):**

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Close and reopen PowerShell, then verify:

```powershell
uv --version
```

**Create and activate an environment:**

```powershell
# In your project folder
uv venv

# Activate (PowerShell)
.venv\Scripts\Activate.ps1

# Activate (Command Prompt)
.venv\Scripts\activate.bat
```

> **Execution policy error?** If PowerShell blocks the activation script, run this once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then retry `.venv\Scripts\Activate.ps1`.

**Install packages:**

```powershell
uv pip install numpy pandas matplotlib seaborn scikit-learn statsmodels jupyter
```

## Optional: WSL 2 (Windows Subsystem for Linux)

WSL gives you a full Linux environment inside Windows. Most data science tooling works more naturally in Linux, so experienced developers often prefer this path.

**Enable WSL:**

```powershell
# Run PowerShell as Administrator
wsl --install
```

Restart your computer when prompted. Ubuntu is installed by default.

**After restart:**

Open **Ubuntu** from the Start menu. Set your Linux username and password, then follow the macOS/Linux path in the standard guides, `curl` for uv, `conda init bash` for Anaconda, etc.

> **VS Code + WSL:** Install the [WSL extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl) and open your project with `code .` from the WSL terminal. VS Code connects to the Linux environment transparently.

## Gotchas

- **`conda` not found in PowerShell**: Anaconda Prompt always works; to also use PowerShell, run `conda init powershell` in Anaconda Prompt, then restart PowerShell.
- **`conda activate` says "run conda init"**: the shell was never initialized. Fix: run `conda init <shell>` (e.g. `conda init powershell`) in Anaconda Prompt, then restart the terminal.
- **Execution policy blocks `.ps1` scripts**: PowerShell may refuse to run `.venv\Scripts\Activate.ps1`. Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` once to allow local scripts.
- **Jupyter opens in the wrong environment**: Navigator apps default to `base`. Use the **Environment** dropdown in Navigator to switch to `dsai` before clicking Launch. From the command line, always `conda activate dsai` first.
- **`Add to PATH` causes conflicts**: checking "Add Anaconda to PATH" during install can shadow system Python and break other tools. The recommended approach is to leave it unchecked and use Anaconda Prompt.
- **Antivirus slows conda/uv significantly**: Windows Defender scanning every file during a large install can make it look frozen. It will eventually finish; if it takes more than 30 minutes, check Windows Security's activity log.
- **OneDrive syncing your project folder**: Python environments inside OneDrive-synced folders (Desktop, Documents) can cause strange errors with file locking. Keep your project folder outside OneDrive (e.g. `C:\Projects\dsai`).
- **Long path names**: some conda packages fail on Windows if the installation path is too long. If you see path-length errors, enable long paths: `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force` (requires Admin PowerShell).

## Verification checklist

Run each line in Anaconda Prompt with `(dsai)` active:

```bash
python --version          # should print Python 3.12.x
conda list | findstr pandas  # should show pandas version
jupyter --version         # should print version without error
```

If all three pass, your Windows environment is ready.

---

**Next step:** [Jupyter Notebook guide](./jupyter-notebook.md) to run your first notebook, or [VS Code guide](./vscode.md) for editor tips.
