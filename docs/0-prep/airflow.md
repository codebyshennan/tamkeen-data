# Getting Started with Apache Airflow

## What is Apache Airflow?

Apache Airflow is a tool that helps you automate and schedule data tasks. Think of it as a smart scheduler for your data work - it can run your Python scripts, SQL queries, and other data processing tasks automatically at specific times (like every day at 2 AM) or when certain conditions are met.

**In simple terms:** Airflow helps you set up automated workflows (called "DAGs") that run your data processing tasks in the right order, handle errors, and send you notifications when things go wrong.

> **Note for beginners:** You don't need to master Airflow right away. Start with the basics and learn as you go. This guide will walk you through everything step by step.

![Airflow Dashboard Placeholder - Shows the Airflow web interface with DAGs list]

## System Requirements

- Python 3.10+ installed
- 4GB RAM minimum (8GB+ recommended)
- 10GB free disk space
- POSIX-compliant operating system (Linux/macOS preferred, Windows via WSL2)

## Installation Options

> **Which option should I choose?**
> - **Option 1 (uv)**: Best for learning and development. Easier to set up and manage.
> - **Option 2 (Docker)**: Better for production environments or if you're already familiar with Docker.

### Option 1: Using uv (Recommended for Beginners)

```bash
# Step 1: Create a new directory for airflow
mkdir airflow
cd airflow

# Step 2: Create and activate virtual environment
# (This keeps Airflow separate from other Python projects)
uv venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Step 3: Set the AIRFLOW_HOME environment variable
# (This tells Airflow where to store its files)
export AIRFLOW_HOME=$(pwd)

# Step 4: Install airflow with minimal dependencies
# (This may take a few minutes - grab a coffee!)
uv pip install apache-airflow

# Step 5: Initialize the database
# (Airflow needs a database to track your workflows)
airflow db init

# Step 6: Create an admin user
# (Replace the email and password with your own)
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

> **What just happened?** You've created a virtual environment (like a separate workspace), installed Airflow, set up its database, and created a user account to access the web interface.

### Option 2: Using Docker (Recommended for Production)

1. Create a new directory:
```bash
mkdir airflow-docker
cd airflow-docker
```

2. Download the docker-compose file:
```bash
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/stable/docker-compose.yaml'
```

3. Create required directories:
```bash
mkdir -p ./dags ./logs ./plugins ./config
```

4. Initialize environment:
```bash
docker-compose up airflow-init
```

5. Start services:
```bash
docker-compose up -d
```

## Initial Configuration

### Core Settings

1. Edit `airflow.cfg` in your AIRFLOW_HOME directory:
```ini
[core]
# Don't load example DAGs
load_examples = False

# Use LocalExecutor for development
executor = LocalExecutor

# Set your timezone
default_timezone = UTC
```

2. Configure database (SQLite is default for development):
```ini
[database]
sql_alchemy_conn = sqlite:///airflow.db
```

### Security Settings

1. Set secure configurations:
```ini
[webserver]
# Enable authentication
authenticate = True

# Use secure connection
web_server_ssl_cert = /path/to/cert
web_server_ssl_key = /path/to/key

# Set session lifetime
session_lifetime_days = 1
```

## Starting Airflow Services

> **Important:** You need to run TWO separate commands in TWO separate terminal windows/tabs. Don't close either one!

### Local Development

**Step 1: Start the Web Server** (Open Terminal Window 1)
```bash
# Make sure you're in your airflow directory and virtual environment is activated
cd airflow
source .venv/bin/activate  # macOS/Linux
# or .venv\Scripts\activate for Windows

# Start the web server
airflow webserver --port 8080
```

You should see output like: `Running the Gunicorn Server with: Workers: 4 threads...`

**Step 2: Start the Scheduler** (Open Terminal Window 2)
```bash
# Navigate to your airflow directory again
cd airflow
source .venv/bin/activate  # macOS/Linux
# or .venv\Scripts\activate for Windows

# Start the scheduler (this is what actually runs your tasks)
airflow scheduler
```

You should see output showing the scheduler is running and checking for DAGs.

**Step 3: Access the Web Interface**

1. Open your web browser
2. Go to: http://localhost:8080
3. Log in with the username and password you created earlier

![Airflow Login Screen Placeholder - Shows the login page]

> **Troubleshooting:** If you can't access the web interface, make sure:
> - The webserver is still running in Terminal 1
> - You're using the correct URL (http://localhost:8080)
> - No other program is using port 8080

### Docker Environment

Monitor services:
```bash
docker-compose ps
```

## Common Issues & Troubleshooting

### Installation Problems

1. **Dependencies Conflict**:
```bash
# Create a fresh virtual environment with uv
uv venv airflow_env
source airflow_env/bin/activate  # Linux/macOS
airflow_env\Scripts\activate     # Windows

# Install with uv (handles constraints automatically)
uv pip install apache-airflow

# Or install specific version if needed
uv pip install "apache-airflow>=2.8.0"
```

2. **Database Issues**:
```bash
# Reset the database
airflow db reset

# Upgrade the database
airflow db upgrade
```

### Runtime Issues

1. **DAGs Not Appearing**:
- Check DAG file permissions
- Verify DAG directory path
- Check for Python syntax errors
- Review airflow logs

2. **Scheduler Not Running**:
```bash
# Check scheduler health
airflow scheduler -- --daemon

# View scheduler logs
tail -f logs/scheduler/latest
```

3. **Worker Problems**:
- Verify executor configuration
- Check resource availability
- Review worker logs

## Best Practices

### DAG Development

**What is a DAG?** DAG stands for "Directed Acyclic Graph" - but don't worry about the technical name! Think of it as a workflow diagram that shows which tasks need to run and in what order.

**Basic DAG Structure:**

```python
# Import necessary libraries
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Define default settings for all tasks in this DAG
default_args = {
    'owner': 'airflow',                    # Who owns this workflow
    'depends_on_past': False,              # Don't wait for previous run to finish
    'start_date': datetime(2025, 1, 1),   # When this DAG should start running
    'email_on_failure': False,             # Don't send emails on failure (for now)
    'retries': 1,                          # Retry once if task fails
    'retry_delay': timedelta(minutes=5),   # Wait 5 minutes before retrying
}

# Create the DAG
with DAG('example_dag',                    # Name of your workflow
         default_args=default_args,        # Use the settings above
         schedule='@daily') as dag:        # Run once per day
    
    # Define your tasks here
    # (We'll add tasks in the next step)
    pass
```

![DAG Example Placeholder - Shows a simple DAG with 3 tasks connected]

2. **Testing**:
- Use `airflow tasks test` for individual task testing
- Implement unit tests for custom operators
- Test DAGs in development environment first

### Production Deployment

1. **Security**:
- Use environment variables for sensitive data
- Implement role-based access control
- Regular security audits

2. **Monitoring**:
- Set up email notifications
- Monitor resource usage
- Regular log review

3. **Scaling**:
- Use CeleryExecutor for distributed tasks
- Configure proper resource pools
- Implement proper retry mechanisms

## Additional Resources

1. **Documentation**:
- [Official Documentation](https://airflow.apache.org/docs/)
- [Best Practices Guide](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)

2. **Community**:
- [GitHub Issues](https://github.com/apache/airflow/issues)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/airflow)
- [Slack Channel](https://apache-airflow.slack.com)

3. **Learning Resources**:
- [Airflow Tutorials](https://airflow.apache.org/docs/apache-airflow/stable/tutorial.html)
- [Example DAGs](https://github.com/apache/airflow/tree/main/airflow/example_dags)
