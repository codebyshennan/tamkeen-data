# DBeaver Setup

<figure><img src="../.gitbook/assets/image (1).png" alt=""><figcaption><p>DBeaver</p></figcaption></figure>

DBeaver Community is a free, open-source universal database tool that helps you manage different database systems through a user-friendly graphical interface. It supports most popular databases including MySQL, PostgreSQL, SQLite, Oracle, and many more.

> **Note:** DBeaver Community is recommended, however you can use any SQL client of your choice for the SQL lessons.

## System Requirements

* Windows 7/8/10/11, macOS 10.13+, or Linux
* 4GB RAM minimum (8GB recommended)
* Java Runtime Environment (JRE) 11+ (bundled with installer)
* 500MB free disk space

## Key Features

* Multiple database support
* SQL editor with syntax highlighting
* Database structure visualization
* Data export/import
* ER diagrams
* SQL execution and debugging

## Installation Instructions

### Download

1. Visit the [DBeaver Community download page](https://dbeaver.io/download/)
2. Choose the installer for your operating system:
   * Windows: Download the .exe installer
   * macOS: Download the .dmg file
   * Linux: Download the appropriate package (.deb for Ubuntu/Debian, .rpm for RHEL/Fedora)

### Installation Steps

{% tabs %}
{% tab title="Windows" %}
* Run the downloaded .exe file
* Follow the installation wizard steps
* Accept the license agreement
* Choose installation location (default is recommended)
* Click "Install" and wait for completion
{% endtab %}

{% tab title="macOS" %}
* Open the downloaded .dmg file
* Drag DBeaver to the Applications folder
* Launch DBeaver from Applications
{% endtab %}
{% endtabs %}

## Initial Setup

### First Launch Configuration

1. Launch DBeaver
2. On first launch, you may be prompted to:
   * Choose a workspace directory (default is fine)
   * Install additional drivers (accept if prompted)
   * Configure proxy settings (if behind corporate firewall)

### Database Setup

{% tabs %}
{% tab title="SQLite (Recommended)" %}
* Click "New Database Connection" button
* Select SQLite
* Choose "Create new database" if you want to start fresh
* Select location and name for your database file
* Test connection and finish
{% endtab %}

{% tab title="PostgresQL" %}
* Click "New Database Connection"
* Select PostgreSQL
* Enter connection details:
  * Host: `localhost` (or your database server)
  * Port: `5432` (default)
  * Database: `your_database_name`
  * Username: `your_username`
  * Password: `your_password`
* Test connection before finishing
{% endtab %}
{% endtabs %}

## Basic Configuration

### Recommended Settings

1. Go to Preferences/Settings:
   * Windows/Linux: Window → Preferences
   * macOS: DBeaver → Preferences
2.  Configure these settings:

    ```
    Editors:
    - Set auto-save interval
    - Enable error highlighting

    SQL Editor:
    - Enable auto-completion
    - Set statement delimiter

    Data Editors:
    - Set fetch size
    - Configure string presentation
    ```

### Security Best Practices

1. **Password Security**:
   * Use "Save Password Locally" with caution
   * Enable master password for stored credentials
2. **SSH Tunneling** (for remote databases):
   * Use SSH tunnel when possible
   * Configure key-based authentication
3. **Network Security**:
   * Use SSL/TLS connections when available
   * Configure timeout settings

## Common Issues & Troubleshooting

### Connection Issues

1. **Cannot Connect to Database**:
   * Verify database is running
   * Check hostname/port
   * Confirm credentials
   * Test network connectivity
   * Check firewall settings
2.  **Driver Issues**:

    ```
    Solutions:
    - Update database driver
    - Download driver manually
    - Clear driver cache
    ```
3. **Performance Problems**:
   * Adjust fetch size
   * Configure result set limits
   * Update database statistics

### Workspace Issues

1. **Slow Performance**:
   * Clear workspace cache
   * Reduce stored connection history
   * Update database statistics
2. **UI Problems**:
   * Reset perspective
   * Clear workspace
   * Update DBeaver

### Error Messages

Common error solutions:

1. "Cannot create driver instance":
   * Reinstall driver
   * Check Java version
2. "Connection refused":
   * Verify database is running
   * Check port number
   * Review firewall settings
3. "Authentication failed":
   * Verify credentials
   * Check database permissions
   * Review authentication method

## Tips & Best Practices

1. **Query Optimization**:
   * Use query explain plan
   * Set appropriate fetch size
   * Use connection pooling
2. **Data Export/Import**:
   * Use native database formats
   * Configure batch sizes
   * Set appropriate timeouts
3. **Version Control**:
   * Save queries as scripts
   * Use project sharing
   * Maintain script templates
