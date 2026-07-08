# Tableau Setup

**After this guide:** Tableau Public is installed, you are signed in, and you can connect a data source and build a simple view or sheet (remember: Public workbooks are visible to others).

## What is Tableau?

Tableau is a powerful tool for creating beautiful, interactive data visualizations (charts, graphs, dashboards) without writing code. Tableau Public is the **free version** that's perfect for learning and sharing your work!

**In simple terms:** Tableau lets you drag and drop your data to create stunning charts and dashboards. Instead of writing code to make graphs, you use a visual interface - like building with blocks!

**Why use Tableau?**
- ✅ **Free** - Tableau Public is completely free
- ✅ **No coding required** - Visual drag-and-drop interface
- ✅ **Beautiful visualizations** - Create professional-looking charts easily
- ✅ **Interactive dashboards** - Make charts that people can explore
- ✅ **Industry standard** - Used by many companies worldwide
- ✅ **Easy sharing** - Publish your work online for free

**Important Note:** Tableau Public workbooks are saved to the cloud and are **publicly visible**. Don't use sensitive or private data!

> **On screen:** Tableau workspace, shelves, marks, data pane.

## Helpful video

Short **Tableau Public** setup: sign up, download, and install (under one minute). Follow the written steps in this guide for account details and security notes.

<iframe width="560" height="315" src="https://www.youtube.com/embed/lTNWfhmurUg" title="Tableau Public Tutorial Download and Setup" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## System Requirements

* Windows 10 or newer / macOS 11.0 (Big Sur) or newer
* 4GB RAM minimum (8GB+ recommended)
* 2GB free disk space
* Internet connection for saving/sharing visualizations

## Key Features

* Create interactive visualizations
* Share work on Tableau Public Gallery
* Connect to various data sources (Excel, CSV, etc.)
* Free cloud storage for public visualizations
* Access to community visualizations for learning
* Built-in data preparation tools

## Setting up Tableau

> **Time needed:** About 15-20 minutes total

### Step 1: Create Account

**Create your free Tableau Public account:**

1. Visit [Tableau Public](https://public.tableau.com)
2. Click **"Sign Up"** in the top right corner
3. Fill out the registration form:
   - **Email:** Use a professional email (your school email works great!)
   - **Password:** Create a strong password (you'll need this to log in)
   - **Username:** Choose a username (this will be visible on your public profile)
   - **Name:** Your display name
4. Check your email and click the verification link
5. Sign in to your new account

> **On screen:** Tableau Public sign-up.

> **Tip:** Your username will be part of your public profile URL, so choose something professional!

### Step 2: Download Tableau

**Download the Tableau Public desktop application:**

1. Visit [Tableau Public Download](https://www.tableau.com/products/public/download)
2. You'll see a form - fill it out (they use this for statistics)
3. Click **"Download the App"** button
4. The download will start automatically

> **On screen:** Download page and installer button.

**Installation:**

<details>
<summary><b>For Windows</b></summary>

1. Find the downloaded **.exe** file (usually in your Downloads folder)
2. Double-click to run the installer
3. If Windows asks for permission (UAC prompt), click **"Yes"**
4. Follow the installation wizard, click **"Next"** through the steps
5. Click **"Install"** and wait 2-5 minutes
6. Click **"Finish"**

**Verify:**

Open **Tableau Public** from the Start menu. The splash screen should appear and prompt you to sign in. If it doesn't launch, try right-clicking the shortcut and choosing **"Run as administrator"** once.

</details>

<details>
<summary><b>For macOS</b></summary>

1. Find the downloaded **.dmg** file (usually in Downloads folder)
2. Double-click to open it
3. Drag the **Tableau Public** icon to the **Applications** folder
4. Open Applications and double-click Tableau Public to launch
5. If macOS asks about security, open **System Settings → Privacy & Security** (or **System Preferences → Security & Privacy** on older macOS) and choose **Open Anyway** for Tableau Public

</details>

> **On screen:** Installer progress.

### Step 3: First Launch Setup

**Configure Tableau when you first open it:**

1. **Accept License Agreement:**
   - Read (or scroll through) the license agreement
   - Click **"Accept"**

2. **Sign In:**
   - Enter the email and password you used to create your account
   - Click **"Sign In"**

> **On screen:** Sign-in to Tableau Public / account.

3. **Welcome Screen:**
   - You might see a welcome tutorial - feel free to skip it for now (you can always access it later)
   - Click **"Get Started"** or close the welcome window

4. **You're ready**
   - You should now see the Tableau workspace
   - The main area is where you'll build your visualizations

> **On screen:** Blank workbook or saved workbook view.

> **Note:** Tableau will automatically save your workbooks to the cloud. You can access them from any computer by logging into your account!

## Data Connection Setup

### Supported File Types

* Excel (.xlsx, .xls)
* Text files (.csv, .txt)
* Google Sheets
* Web data connectors
* Spatial files (.kml, .geojson)

### Connecting to Data

**Step 1: Start a New Connection**
1. When you open Tableau, you'll see a "Connect" panel on the left
2. Under "To a File," you'll see options like:
   - **Microsoft Excel**: For **.xlsx** or **.xls** files
   - **Text File**: For **.csv** or **.txt** files
   - **More...** - For other file types

> **On screen:** Connect to data, file / server options.

**Step 2: Choose Your Data File**
1. Click on the file type you want (e.g., "Microsoft Excel" or "Text File")
2. A file browser will open
3. Navigate to where your data file is saved
4. Select your file and click **"Open"**

**Step 3: Preview and Prepare Your Data**
1. You'll see a preview of your data in the bottom panel
2. You can:
   - **Rename fields** - Double-click a column name to rename it
   - **Change data types** - Click the data type icon (Abc, #, calendar) to change it
   - **Split columns** - Right-click a column for options
   - **Remove columns** - Uncheck columns you don't need

> **On screen:** Data source preview, fields and types.

**Step 4: Load Your Data**
1. Once you're happy with your data, click **"Sheet 1"** at the bottom
2. Your data is now loaded and ready to use!

> **Tip:** Tableau is smart about detecting data types, but always double-check that numbers are recognized as numbers (not text) and dates are recognized as dates!

### Best Practices

1. **Data Preparation**:
   * Clean data before importing
   * Use consistent naming conventions
   * Remove unnecessary columns
2. **Performance**:
   * Keep file sizes manageable
   * Use extracts for large datasets
   * Index important columns

## Gotchas

- **Tableau Public = public**: every workbook you save is visible to anyone on the internet. Never upload data that includes personal information, confidential figures, or anything your employer considers sensitive.
- **No live database connections**: Tableau Public can only connect to files (Excel, CSV, Google Sheets, spatial files). It cannot connect directly to PostgreSQL, Snowflake, or other live databases. For live-database exercises, use Tableau Desktop (14-day trial) or export a query result to CSV first.
- **Saving requires an internet connection and login**: Tableau Public saves to the cloud, not to a local file. If your internet drops mid-save, the workbook is not saved. Use **File → Export Packaged Workbook (.twbx)** to keep a local backup.
- **`.twb` vs `.twbx`**: a `.twb` file references an external data source but does not embed the data. A `.twbx` (packaged workbook) embeds the data. For sharing or reopening on another machine, always use `.twbx` or Tableau Public.
- **macOS security warning on first launch**: if macOS says Tableau Public can't be opened, go to **System Settings → Privacy & Security → Open Anyway**. This is standard for apps distributed outside the App Store.
- **Large files are slow**: Tableau Public works best with files under ~1 million rows. If your CSV is large, pre-aggregate it in pandas or SQL before connecting.
- **Field type detection errors**: Tableau auto-detects data types, but it sometimes reads numeric IDs as numbers and tries to SUM them. Right-click the field in the Data pane and change the role to **Dimension** (or type to **String**) when this happens.



### Installation Problems

1. **Installation Fails**:
   * Run as administrator
   * Check system requirements
   * Clear temporary files
   * Disable antivirus temporarily
2. **Application Won't Start**:
   * Clear Tableau cache
   * Reinstall application
   * Check firewall settings

### Data Connection Issues

1. **Can't Connect to File**:
   * Verify file format
   * Check file permissions
   * Move file to local drive
2. **Slow Performance**:
   * Create data extract
   * Reduce dataset size
   * Close unnecessary worksheets

### Publishing Problems

1. **Upload Fails**:
   * Check internet connection
   * Reduce workbook size
   * Clear browser cache
   * Try different browser
2. **Visualization Not Updating**:
   * Refresh browser
   * Clear browser cache
   * Check data source refresh

## Tips & Best Practices

### Workspace Organization

1. **Project Structure**:
   * Use clear naming conventions
   * Organize related sheets in dashboards
   * Keep similar visualizations together
2. **Performance Optimization**:
   * Hide unused fields
   * Use filters efficiently
   * Limit number of marks

### Data Management

1. **Source Files**:
   * Keep backup copies
   * Document data sources
   * Update regularly
2. **Security**:
   * Remove sensitive data
   * Check sharing settings
   * Review before publishing

### Visualization Design

1. **Best Practices**:
   * Start with clear objectives
   * Choose appropriate chart types
   * Use consistent formatting
2. **Interactivity**:
   * Add meaningful filters
   * Include tooltips
   * Use action filters

## Additional Resources

1. **Learning**:
   * [Tableau Public Gallery](https://public.tableau.com/gallery)
   * [Tableau Learning Center](https://www.tableau.com/learn)
   * Community Forums
2. **Support**:
   * [Knowledge Base](https://www.tableau.com/support/knowledgebase)
   * Community Forums
   * Video Tutorials
