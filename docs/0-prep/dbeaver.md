# DBeaver Setup

## What is DBeaver?

DBeaver is a **free** database tool that lets you connect to and work with databases using a friendly graphical interface. Instead of typing commands in a terminal, you get buttons, menus, and visual tools to explore your data!

**In simple terms:** DBeaver is like a "file explorer" for databases. You can:
- Browse tables and data visually
- Write and run SQL queries
- See your data in easy-to-read tables
- Export data to Excel or CSV files
- Create diagrams showing how tables connect

**Why use DBeaver?**
- ✅ **Free** - Community edition is completely free
- ✅ **Works with many databases** - MySQL, PostgreSQL, SQLite, Snowflake, and more
- ✅ **Visual interface** - No need to remember complex commands
- ✅ **Great for beginners** - Easy to learn and use
- ✅ **Powerful** - Has everything you need for SQL work

![DBeaver Interface Placeholder - Shows the main DBeaver window with database connections]

> **Note:** DBeaver Community is recommended for this course, but you can use any SQL client you prefer. DBeaver is just the easiest to get started with!

## System Requirements

* Windows 10/11, macOS 10.15+, or Linux
* 4GB RAM minimum (8GB recommended)
* Java Runtime Environment (JRE) 17+ (bundled with installer)
* 500MB free disk space

## Key Features

* Multiple database support
* SQL editor with syntax highlighting
* Database structure visualization
* Data export/import
* ER diagrams
* SQL execution and debugging

## Installation Instructions

> **Time needed:** About 5-10 minutes

### Step 1: Download DBeaver

1. Visit the [DBeaver Community download page](https://dbeaver.io/download/)
2. You'll see different versions - choose **"Community Edition"** (it's free!)
3. Click the download button for your operating system:
   * **Windows:** Download the `.exe` installer
   * **macOS:** Download the `.dmg` file
   * **Linux:** Download the appropriate package (`.deb` for Ubuntu/Debian, `.rpm` for RHEL/Fedora)

![DBeaver Download Page Placeholder - Shows the download options]

### Step 2: Install DBeaver

<details>
<summary><b>Windows Installation</b></summary>

1. Find the downloaded `.exe` file (usually in your Downloads folder)
2. Double-click to run the installer
3. If Windows asks for permission, click **"Yes"**
4. Follow the installation wizard:
   - Accept the license agreement
   - Choose installation location (default is fine - just click "Next")
   - Click **"Install"** and wait for it to finish
5. Click **"Finish"** when done
6. DBeaver should launch automatically!

![Windows Installation Wizard Placeholder - Shows the installation steps]

</details>

<details>
<summary><b>macOS Installation</b></summary>

1. Find the downloaded `.dmg` file (usually in your Downloads folder)
2. Double-click to open it
3. You'll see a window with the DBeaver icon
4. Drag the **DBeaver** icon to the **Applications** folder
5. Wait for the copy to complete
6. Open **Applications** folder and double-click **DBeaver** to launch
7. If macOS asks about security, go to **System Preferences → Security & Privacy** and click **"Open Anyway"**

![macOS DMG Window Placeholder - Shows dragging DBeaver to Applications]

</details>

## Initial Setup

### First Launch Configuration

**Step 1: Launch DBeaver**
- **Windows:** Find DBeaver in your Start menu and click it
- **macOS:** Open Applications folder and double-click DBeaver

**Step 2: Initial Setup Prompts**

When you first open DBeaver, you might see a few popups:

1. **Workspace Directory:**
   - DBeaver will ask where to save your projects
   - **Just click "OK"** - the default location is perfect!
   - This is where DBeaver saves your database connections and settings

![Workspace Selection Dialog Placeholder - Shows the workspace directory prompt]

2. **Install Additional Drivers:**
   - DBeaver might ask to download database drivers
   - **Click "Yes" or "Download"** - these are needed to connect to databases
   - This might take a minute or two

3. **Proxy Settings:**
   - Only appears if you're behind a corporate firewall
   - Most students can skip this
   - If you're at school/work and having connection issues, ask your IT department

> **Tip:** Don't worry if you're not sure what to choose - the defaults work great for learning!

### Database Setup

**What is a database connection?** It's like adding a bookmark - you're telling DBeaver how to connect to a specific database so you can work with it.

**Step 1: Create a New Connection**

1. Look for the **"New Database Connection"** button in the toolbar (it looks like a plug or database icon)
   - OR click **"Database"** in the menu bar → **"New Database Connection"**
2. A window will pop up showing different database types

![New Connection Dialog Placeholder - Shows the database selection screen]

<details>
<summary><b>SQLite (Recommended for Beginners)</b></summary>

**SQLite is perfect for learning!** It's a simple database that stores everything in a single file on your computer.

1. In the connection window, find and click **"SQLite"**
2. Click **"Next"**
3. You'll see options:
   - **Path:** Click the folder icon to choose where to save your database file
   - **Database name:** Give it a name like "my_first_database.db"
4. Click **"Test Connection"** - you should see "Connected" in green
5. Click **"Finish"**

![SQLite Connection Setup Placeholder - Shows SQLite connection configuration]

> **Tip:** SQLite databases are just files on your computer - you can easily copy, backup, or delete them!

</details>

<details>
<summary><b>PostgreSQL (If You Have a Server)</b></summary>

**PostgreSQL** is a more advanced database that runs on a server. You'll need connection details from your instructor or database administrator.

1. In the connection window, find and click **"PostgreSQL"**
2. Click **"Next"**
3. Fill in the connection details:
   - **Host:** Usually `localhost` (if on your computer) or an IP address
   - **Port:** Usually `5432` (the default)
   - **Database:** The name of the database you want to connect to
   - **Username:** Your database username
   - **Password:** Your database password
4. Click **"Test Connection"** - if successful, you'll see "Connected"
5. Click **"Finish"**

![PostgreSQL Connection Setup Placeholder - Shows PostgreSQL connection form]

> **Note:** If you don't have a PostgreSQL server set up, start with SQLite instead!

</details>

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
