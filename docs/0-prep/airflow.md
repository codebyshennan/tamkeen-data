# Getting Started with Apache Airflow

Apache Airflow is a platform to programmatically author, schedule, and monitor workflows. It's essential for data engineering tasks and ETL processes.

## System Requirements

- Python 3.8+ installed
- 4GB RAM minimum (8GB+ recommended)
- 10GB free disk space
- POSIX-compliant operating system (Linux/macOS preferred, Windows via WSL2)

## Installation Options

### Option 1: Using pip (Recommended for Development)

```bash
# Create a new directory for airflow
mkdir airflow
cd airflow

# Set the AIRFLOW_HOME environment variable
export AIRFLOW_HOME=$(pwd)

# Install airflow with minimal dependencies
pip install apache-airflow

# Initialize the database
airflow db init

# Create an admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

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

### Local Development

1. Start the webserver:
```bash
airflow webserver --port 8080
```

2. In a new terminal, start the scheduler:
```bash
airflow scheduler
```

3. Access the UI at http://localhost:8080

### Docker Environment

Monitor services:
```bash
docker-compose ps
```

## Common Issues & Troubleshooting

### Installation Problems

1. **Dependencies Conflict**:
```bash
# Create a fresh virtual environment
python -m venv airflow_env
source airflow_env/bin/activate  # Linux/macOS
airflow_env\Scripts\activate     # Windows

# Install with constraints
pip install apache-airflow==2.7.1 --constraint constraints-3.8.txt
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

1. **Structure**:
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG('example_dag',
         default_args=default_args,
         schedule_interval='@daily') as dag:
    
    # Define tasks here
    pass
```

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
