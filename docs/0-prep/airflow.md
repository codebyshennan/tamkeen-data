# Getting Started with Apache Airflow

**After this guide:** Airflow runs on your machine (or via Docker as described below), you can open the **web UI** (for example `http://localhost:8080`), sign in, and see DAGs, enough to follow orchestration labs.

## What is Apache Airflow?

Apache Airflow is a tool that helps you automate and schedule data tasks. Think of it as a smart scheduler for your data work - it can run your Python scripts, SQL queries, and other data processing tasks automatically at specific times (like every day at 2 AM) or when certain conditions are met.

**In simple terms:** Airflow helps you set up automated workflows (called "DAGs") that run your data processing tasks in the right order, handle errors, and send you notifications when things go wrong.

> **Note for beginners:** You don't need to master Airflow right away. Start with the basics and learn as you go. This guide will walk you through everything step by step.

> **On screen:** Airflow web UI, DAG list / home dashboard.

## Helpful video

High-level introduction to **DAGs**, tasks, operators, scheduling, and the web UI, pairs well with the install steps below.

<iframe width="560" height="315" src="https://www.youtube.com/embed/eeSLDdz-aLg" title="Apache Airflow Tutorial for Beginners: Workflow Orchestration Explained" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## System Requirements

- **Python 3.10, 3.11, 3.12, or 3.13** (Airflow 3.1.x does not support Python 3.9 or earlier, and 3.14+ is not supported yet, see the [supported versions](https://airflow.apache.org/docs/apache-airflow/stable/installation/supported-versions.html) page for your install date)
- 4GB RAM minimum (8GB+ recommended)
- 10GB free disk space
- POSIX-compliant operating system (Linux/macOS preferred, Windows via WSL2)

> **Windows users:** Airflow's install scripts use bash syntax that doesn't work in PowerShell or cmd. The easiest path is **WSL2** (Windows Subsystem for Linux), which gives you a real Linux environment. See the [Windows setup guide](./windows.md) for WSL2 install steps, then follow Option 1 below inside your WSL2 terminal. A native PowerShell option is also documented below if you cannot use WSL2.

Airflow is pinned to **tested dependency sets** via official **constraints** files. A plain `pip install apache-airflow` or `uv pip install apache-airflow` without constraints often fails or yields a broken install, use the commands below.

## Installation Options

> **Which option should I choose?**
>
> - **Option 1 (uv + constraints)**: Matches the [official Quick Start](https://airflow.apache.org/docs/apache-airflow/stable/start.html). Best for learning on your laptop.
> - **Option 2 (Docker)**: Closer to how teams run Airflow in production; use if you already use Docker.

### Option 1: Using uv (recommended)

```bash
# Step 1: Create a new directory for Airflow (keep it separate from your course "dsai" env)
mkdir airflow-local
cd airflow-local

# Step 2: Create and activate a virtual environment
uv venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Step 3: Set AIRFLOW_HOME (where config, logs, and the DB live)
export AIRFLOW_HOME="$(pwd)/airflow_home"   # macOS/Linux
# set AIRFLOW_HOME=%CD%\airflow_home        # Windows CMD (example)

# Step 4: Install a pinned Airflow release using the official constraints file for your Python version
AIRFLOW_VERSION=3.1.8
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
uv pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
```

**Easiest way to run everything (database, admin user, scheduler, UI):**

```bash
airflow standalone
```

Open **<http://localhost:8080>**, sign in with the **admin** credentials printed in the terminal, and enable an example DAG from the home page if you like.

> **What just happened?** `standalone` initializes the metadata database, creates an admin user, and starts the processes needed for local development. When you outgrow it, run components separately (see below).

**If you prefer to start services yourself** (no `standalone`):

```bash
airflow db migrate

airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Airflow 3.x serves the web UI via the API server (not "airflow webserver")
airflow api-server --port 8080
```

In **additional terminals** (same `AIRFLOW_HOME` and venv), run:

```bash
airflow scheduler
airflow dag-processor
airflow triggerer
```

The `users create` flow requires the [FAB auth manager](https://airflow.apache.org/docs/apache-airflow-providers-fab/stable/auth-manager/index.html) as in a default install; if your org uses a different auth setup, follow your administrator's docs.

### Option 1b: Windows: PowerShell (if you cannot use WSL2)

The bash multi-line variable expansion in Option 1 doesn't work in PowerShell. Use the equivalent PowerShell commands instead:

```powershell
# Step 1: Create and enter a new directory
mkdir airflow-local
cd airflow-local

# Step 2: Create and activate a virtual environment
uv venv
.venv\Scripts\Activate.ps1
# If blocked by execution policy, run once: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# Step 3: Set AIRFLOW_HOME for this session
$env:AIRFLOW_HOME = "$PWD\airflow_home"

# Step 4: Install Airflow with the official constraint file
$AIRFLOW_VERSION = "3.1.8"
$PYTHON_VERSION = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$CONSTRAINT_URL = "https://raw.githubusercontent.com/apache/airflow/constraints-$AIRFLOW_VERSION/constraints-$PYTHON_VERSION.txt"
uv pip install "apache-airflow==$AIRFLOW_VERSION" --constraint $CONSTRAINT_URL
```

> **Make `AIRFLOW_HOME` permanent:** PowerShell `$env:` variables reset when the terminal closes. To persist it, add a line to your PowerShell profile (`notepad $PROFILE`):
> ```powershell
> $env:AIRFLOW_HOME = "C:\Users\YourName\airflow-local\airflow_home"
> ```

**Start Airflow:**

```powershell
airflow standalone
```

Open `http://localhost:8080` in your browser. Sign in with the admin credentials printed in the terminal.

> **Known limitation:** Some Airflow features (file watchers, Unix sockets) behave differently on Windows. For production-like testing, WSL2 is still recommended. For learning DAGs and the web UI, PowerShell is fine.

### Option 2: Using Docker

1. Create a new directory:

```bash
mkdir airflow-docker
cd airflow-docker
```

1. Download the docker-compose file:

```bash
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/stable/docker-compose.yaml'
```

1. Create required directories:

```bash
mkdir -p ./dags ./logs ./plugins ./config
```

1. Initialize environment (Compose v2 plugin syntax):

```bash
docker compose up airflow-init
```

1. Start services:

```bash
docker compose up -d
```

If your machine still has the old `docker-compose` binary, that works too; new Docker installs use **`docker compose`** (with a space).

## Initial Configuration

Settings below are typical for **local learning** installs. **Airflow 3.x** uses an **API server** for the web UI and may rename or relocate some options, confirm names in [Configuration reference](https://airflow.apache.org/docs/apache-airflow/stable/configurations-ref.html) for your exact version.

### Core Settings

1. Edit **airflow.cfg** in your **AIRFLOW_HOME** directory (created after the first `airflow` command):

```ini
[core]
# Don't load example DAGs
load_examples = False

# Use LocalExecutor for development
executor = LocalExecutor

# Set your timezone
default_timezone = UTC
```

1. Configure database (SQLite is default for development):

```ini
[database]
sql_alchemy_conn = sqlite:///airflow.db
```

### Security Settings

For production or shared networks, follow [Production deployment](https://airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/production-deployment.html) and TLS guidance for the **API server / web UI** in your Airflow version's docs. Local `standalone` installs use defaults suitable for **localhost only**.

## Starting Airflow Services

> **Important:** You need to run TWO separate commands in TWO separate terminal windows/tabs. Don't close either one!

### Local Development

**Option A, one command (simplest):**

<details>
<summary><b>macOS / Linux / WSL2</b></summary>

```bash
cd airflow-local
source .venv/bin/activate
export AIRFLOW_HOME=...   # same as when you installed
airflow standalone
```

</details>

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
cd airflow-local
.venv\Scripts\Activate.ps1
$env:AIRFLOW_HOME = "$PWD\airflow_home"
airflow standalone
```

</details>

**Option B, separate processes** (typical when you have already run `airflow db migrate` and created a user):

**Terminal 1, API server (web UI):**

<details>
<summary><b>macOS / Linux / WSL2</b></summary>

```bash
cd airflow-local
source .venv/bin/activate
airflow api-server --port 8080
```

</details>

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
cd airflow-local
.venv\Scripts\Activate.ps1
$env:AIRFLOW_HOME = "$PWD\airflow_home"
airflow api-server --port 8080
```

</details>

**Terminal 2, scheduler:**

<details>
<summary><b>macOS / Linux / WSL2</b></summary>

```bash
cd airflow-local
source .venv/bin/activate
airflow scheduler
```

</details>

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
cd airflow-local
.venv\Scripts\Activate.ps1
$env:AIRFLOW_HOME = "$PWD\airflow_home"
airflow scheduler
```

</details>

**Terminal 3, DAG processor and triggerer** (required in multi-process setups; often started for you by `standalone`):

```bash
airflow dag-processor
airflow triggerer
```

**Step 3: Access the Web Interface**

1. Open your web browser
2. Go to: <http://localhost:8080>
3. Log in with the username and password you created earlier

> **On screen:** Airflow login page (**<http://localhost:8080>**).

> **Troubleshooting:** If you can't access the web interface, make sure:
>
> - **`airflow standalone`** is still running, or the **API server** (`airflow api-server`) is still running in its terminal
> - You're using the correct URL (<http://localhost:8080>)
> - No other program is using port 8080

### Docker Environment

Monitor services:

```bash
docker compose ps
```

## Gotchas

- **Windows: `AIRFLOW_HOME` resets every terminal**: PowerShell `$env:AIRFLOW_HOME` only lasts for the current session. If you open a new terminal and forget to set it, Airflow initializes a fresh instance in `~/airflow`. Fix: add `$env:AIRFLOW_HOME = "C:\...\airflow_home"` to your PowerShell profile (`$PROFILE`), or use WSL2 where `export` in `.bashrc` persists.
- **Windows: bash install script won't run in PowerShell or cmd**: the multi-line variable expansion in Option 1 uses bash syntax. Use Option 1b (PowerShell) or WSL2.
- **Never install without constraint files**: plain `pip install apache-airflow` or `uv pip install apache-airflow` almost always produces a broken install due to conflicting transitive dependencies. Always use the official `--constraint` URL for your Airflow version and Python version.
- **`AIRFLOW_HOME` must be consistent**: Airflow stores its database, config, and DAGs relative to `AIRFLOW_HOME`. If you open a new terminal and forget to set it, `airflow standalone` initializes a *fresh* second Airflow instance in `~/airflow`, and your DAGs won't appear. Set `AIRFLOW_HOME` in your shell profile or always `cd` to the project folder and export before running Airflow.
- **Airflow 3.x uses `airflow api-server`, not `airflow webserver`**: guides written for Airflow 2.x say `airflow webserver`. In Airflow 3.x, the web UI is served by `airflow api-server --port 8080`. Using the old command will fail or start nothing.
- **DAGs don't appear immediately**: the scheduler picks up new DAG files every 30-60 seconds (configurable). If your DAG is missing, wait a minute, then check `airflow dags list` for import errors (a syntax error in the Python file will silently prevent loading).
- **`load_examples = True` by default**: Airflow ships with ~20 example DAGs. They clutter the UI when learning. Set `load_examples = False` in `airflow.cfg` and run `airflow db migrate` to remove them.
- **SQLite only supports one active process**: with `LocalExecutor` or `CeleryExecutor`, multiple workers try to write the SQLite database concurrently and will fail. SQLite is only suitable for `SequentialExecutor` (one task at a time), which is the default for learning. Upgrade to PostgreSQL when you need parallelism.
- **Port 8080 already in use**: if something else is on port 8080 (another Airflow, Jupyter, a local server), the API server won't start. Change the port with `airflow api-server --port 8090` or stop the conflicting process first.



### Installation Problems

1. **Dependencies Conflict**:

```bash
# Create a fresh virtual environment with uv
uv venv airflow_env
source airflow_env/bin/activate  # Linux/macOS
airflow_env\Scripts\activate     # Windows

# Install with uv using official constraints (replace AIRFLOW_VERSION and use your Python minor version)
AIRFLOW_VERSION=3.1.8
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
uv pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
```

1. **Database Issues**:

```bash
# Apply migrations (preferred over legacy init)
airflow db migrate

# Reset the database (destructive)
airflow db reset
```

### Runtime Issues

1. **DAGs Not Appearing**:

- Check DAG file permissions
- Verify DAG directory path
- Check for Python syntax errors
- Review airflow logs

1. **Scheduler Not Running**:

```bash
# Check scheduler health
airflow scheduler -- --daemon

# View scheduler logs
tail -f logs/scheduler/latest
```

1. **Worker Problems**:

- Verify executor configuration
- Check resource availability
- Review worker logs

## Best Practices

### DAG Development

**What is a DAG?** DAG stands for "Directed Acyclic Graph" - but don't worry about the technical name! Think of it as a workflow diagram that shows which tasks need to run and in what order.

**Basic DAG structure (illustrative):** Airflow 3.x continues to use DAGs and operators, but defaults and imports evolve by version. Use the [current tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/index.html) as the source of truth.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "example_dag",
    default_args=default_args,
    schedule="@daily",
) as dag:
    pass  # add tasks here
```

> **On screen:** Graph view of a simple DAG (2-3 tasks) in the UI.

1. **Testing**:

- Use **`airflow tasks test`** for individual task testing (see the CLI help for your installed version)
- Implement unit tests for custom operators
- Test DAGs in development environment first

### Production Deployment

1. **Security**:

- Use environment variables for sensitive data
- Implement role-based access control
- Regular security audits

1. **Monitoring**:

- Set up email notifications
- Monitor resource usage
- Regular log review

1. **Scaling**:

- Use CeleryExecutor for distributed tasks
- Configure proper resource pools
- Implement proper retry mechanisms

## Additional Resources

1. **Documentation**:

- [Official Documentation](https://airflow.apache.org/docs/)
- [Best Practices Guide](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)

1. **Community**:

- [GitHub Issues](https://github.com/apache/airflow/issues)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/airflow)
- [Slack Channel](https://apache-airflow.slack.com)

1. **Learning Resources**:

- [Airflow Tutorials](https://airflow.apache.org/docs/apache-airflow/stable/tutorial.html)
- [Example DAGs](https://github.com/apache/airflow/tree/main/airflow/example_dags)
