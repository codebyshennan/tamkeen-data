# 🌐 GitHub Pages Deployment Guide

## Current Setup Overview

Your project uses **Jekyll with GitHub Pages** to create a static educational website.

**Live Site:** https://codebyshennan.github.io/tamkeen-data

---

## 📐 Architecture

```
Repository: codebyshennan/tamkeen-data
    ↓
docs/ folder (Jekyll source)
    ↓
GitHub Pages Build
    ↓
Static Website: codebyshennan.github.io/tamkeen-data
```

---

## ⚙️ How It Works

### 1. **Jekyll Configuration** (`_config.yml`)

```yaml
title: Data Science and AI Course
baseurl: "/tamkeen-data"        # ← Subfolder on GitHub Pages
url: "https://codebyshennan.github.io"  # ← Your GitHub Pages domain
repository: codebyshennan/tamkeen-data  # ← GitHub repo

# Theme
remote_theme: pages-themes/primer@v0.6.0  # ← Uses GitHub's Primer theme

# Markdown Processing
markdown: kramdown
highlighter: rouge
kramdown:
  math_engine: mathjax           # ← Math formulas support
  syntax_highlighter: rouge      # ← Code highlighting
  syntax_highlighter_opts:
    default_lang: python         # ← Python is default language
```

### 2. **Content Structure**

```
docs/
├── _config.yml              # Jekyll configuration
├── index.md                 # Homepage
├── 0-prep/                  # Module 0 (EXCLUDED from build)
├── 1-data-fundamentals/     # Module 1 (EXCLUDED from build)
│   └── 1.2-intro-python/    # Your enhanced Python materials
│       ├── README.md
│       ├── data-structures.md
│       ├── video-resources.md
│       └── notebooks/
├── 2-data-wrangling/        # Module 2
├── 3-visualization/         # Module 3
└── ...
```

**⚠️ Important:** Currently `0-prep/` and `1-data-fundamentals/` are **EXCLUDED** from the site build (see lines 58-61 in `_config.yml`)

```yaml
exclude:
  - 0-prep/
  - 1-data-fundamentals/
  - docs/0-prep/
  - docs/1-data-fundamentals/
```

This means your enhanced Python materials **won't appear on the live site** unless you remove these exclusions!

---

## 🚀 Deployment Methods

Your project has **two deployment options** (currently GitHub Actions is disabled):

### Method 1: Automatic GitHub Actions (Currently Disabled)

**File:** `.github/workflows/jekyll-gh-pages.yml.disabled`

**How it would work:**
1. You push changes to `main` branch
2. GitHub Actions automatically:
   - Sets up Ruby and Jekyll
   - Builds the site from `docs/` folder
   - Deploys to GitHub Pages
3. Site is live in ~2-5 minutes

**To enable this:**
```bash
cd /Users/wongshennan/Documents/work/skillsunion/dsai/tamkeen
mv .github/workflows/jekyll-gh-pages.yml.disabled .github/workflows/jekyll-gh-pages.yml
git add .github/workflows/jekyll-gh-pages.yml
git commit -m "Enable GitHub Actions deployment"
git push
```

**Then configure on GitHub:**
1. Go to: https://github.com/codebyshennan/tamkeen-data/settings/pages
2. Under "Build and deployment"
3. Source: Choose "GitHub Actions"
4. Save

### Method 2: Manual GitHub Pages (Likely Current Method)

**Current settings (probably):**
- Source: Deploy from a branch
- Branch: `main` or `gh-pages`
- Folder: `/docs` or `/` (root)

**To check/configure:**
1. Visit: https://github.com/codebyshennan/tamkeen-data/settings/pages
2. Look at current "Source" setting
3. GitHub Pages automatically builds Jekyll sites

---

## 📝 How to Update the Live Site

### Quick Update Process

```bash
# 1. Make your changes (like the Python enhancements you just did)
cd /Users/wongshennan/Documents/work/skillsunion/dsai/tamkeen

# 2. Commit changes
git add docs/1-data-fundamentals/1.2-intro-python/
git commit -m "Enhance Python intro materials with detailed explanations"

# 3. Push to GitHub
git push origin main

# 4. Wait 2-5 minutes for GitHub Pages to rebuild

# 5. Check your site
open https://codebyshennan.github.io/tamkeen-data
```

---

## 🔧 Making Your Python Materials Visible

Your enhanced materials are currently excluded! To include them:

### Option 1: Remove Exclusions (Recommended)

Edit `docs/_config.yml`:

```yaml
# Remove or comment out these lines:
exclude:
  - .git/
  - .gitbook/
  - node_modules/
  # - 0-prep/                    # ← Remove this
  # - 1-data-fundamentals/       # ← Remove this
  # - docs/0-prep/               # ← Remove this
  # - docs/1-data-fundamentals/  # ← Remove this
  - Gemfile
  - Gemfile.lock
  - vendor/
```

### Option 2: Include Only Specific Modules

Add to `_config.yml`:

```yaml
include:
  - 1-data-fundamentals/1.2-intro-python/
```

---

## 🎯 URL Structure

After deployment, your materials will be accessible at:

```
Base URL: https://codebyshennan.github.io/tamkeen-data/

Your Python materials:
├─ /1-data-fundamentals/1.2-intro-python/README.html
├─ /1-data-fundamentals/1.2-intro-python/data-structures.html
├─ /1-data-fundamentals/1.2-intro-python/video-resources.html
└─ /1-data-fundamentals/1.2-intro-python/notebooks/
```

**Full URLs:**
- Main: `https://codebyshennan.github.io/tamkeen-data/1-data-fundamentals/1.2-intro-python/`
- Data Structures: `https://codebyshennan.github.io/tamkeen-data/1-data-fundamentals/1.2-intro-python/data-structures`
- Videos: `https://codebyshennan.github.io/tamkeen-data/1-data-fundamentals/1.2-intro-python/video-resources`

---

## 🧪 Testing Locally Before Deployment

### Method 1: Using Jekyll (If you set up Gemfile)

```bash
cd /Users/wongshennan/Documents/work/skillsunion/dsai/tamkeen/docs

# Install dependencies (first time only)
bundle install

# Serve locally
bundle exec jekyll serve

# Open in browser
open http://localhost:4000/tamkeen-data/
```

### Method 2: Simple Python Server (Quick Preview)

```bash
cd /Users/wongshennan/Documents/work/skillsunion/dsai/tamkeen/docs
python3 -m http.server 8000

# Open in browser
open http://localhost:8000/1-data-fundamentals/1.2-intro-python/
```

**Note:** This won't process Jekyll (no theme, no includes), but shows raw content.

### Method 3: Markdown Preview in Cursor

- Open any `.md` file
- Press `Cmd+Shift+V` (Mac) or `Ctrl+Shift+V` (Windows)
- See rendered markdown (no Jekyll theme, but good for content review)

---

## 🎨 How Jekyll Processes Your Content

### 1. Markdown → HTML Conversion

```markdown
# Your markdown file (data-structures.md)
## Introduction to Data Structures
Some text here...
```

Jekyll converts to:

```html
<!-- data-structures.html -->
<h1>Data Structures for Data Analysis</h1>
<h2>Introduction to Data Structures</h2>
<p>Some text here...</p>
```

### 2. Theme Application

Jekyll applies the Primer theme:
- Navigation bar
- Styling (colors, fonts)
- Layout (header, content, footer)
- Responsive design

### 3. Code Highlighting

```python
# Your Python code in markdown
def hello():
    print("Hello, World!")
```

Gets syntax highlighted with Rouge:

```html
<div class="highlight">
  <pre class="highlight python">
    <code><span class="k">def</span> <span class="nf">hello</span><span class="p">():</span>
      <span class="k">print</span><span class="p">(</span><span class="s">"Hello, World!"</span><span class="p">)</span>
    </code>
  </pre>
</div>
```

### 4. Math Rendering (MathJax)

```markdown
$$E = mc^2$$
```

Becomes interactive math:
$$E = mc^2$$

---

## 📋 Complete Deployment Checklist

To make your enhanced Python materials live:

### Step 1: Update Configuration
```bash
# Edit docs/_config.yml
# Remove exclusions for 1-data-fundamentals/
```

### Step 2: Test Locally (Optional)
```bash
cd docs
bundle exec jekyll serve
# Visit http://localhost:4000/tamkeen-data/
```

### Step 3: Commit and Push
```bash
git add docs/1-data-fundamentals/
git add docs/_config.yml
git commit -m "Add enhanced Python materials to site"
git push origin main
```

### Step 4: Enable GitHub Actions (Recommended)
```bash
mv .github/workflows/jekyll-gh-pages.yml.disabled \
   .github/workflows/jekyll-gh-pages.yml
git add .github/workflows/
git commit -m "Enable GitHub Actions deployment"
git push
```

### Step 5: Configure GitHub Pages
1. Go to: https://github.com/codebyshennan/tamkeen-data/settings/pages
2. Source: Select "GitHub Actions"
3. Wait for deployment (~2-5 minutes)

### Step 6: Verify
```bash
open https://codebyshennan.github.io/tamkeen-data/1-data-fundamentals/1.2-intro-python/
```

---

## 🔍 Troubleshooting

### Issue: Changes don't appear on site

**Solutions:**
1. Check if folder is excluded in `_config.yml`
2. Clear browser cache (Cmd+Shift+R)
3. Wait 5 minutes (GitHub Pages rebuild takes time)
4. Check GitHub Actions tab for build errors

### Issue: Broken links

**Cause:** Incorrect baseurl in links

**Fix:** Use relative links:
```markdown
<!-- ✅ Good -->
[Link to videos](./video-resources.md)

<!-- ❌ Bad -->
[Link to videos](https://codebyshennan.github.io/tamkeen-data/1-data-fundamentals/1.2-intro-python/video-resources)
```

### Issue: Math formulas not rendering

**Fix:** Ensure MathJax is configured in `_config.yml`:
```yaml
kramdown:
  math_engine: mathjax
```

### Issue: Code not highlighted

**Fix:** Check syntax specifier:
```markdown
<!-- ✅ Good -->
```python
def hello():
    pass
```

<!-- ❌ Bad -->
```
def hello():
    pass
```
```

---

## 📊 Build Process Visualization

```
┌─────────────────────────────────────────────┐
│  Local Changes                              │
│  ├─ Edit data-structures.md                │
│  ├─ Add video-resources.md                 │
│  └─ Create notebooks/                      │
└────────────────┬────────────────────────────┘
                 │
                 ↓ git push
┌─────────────────────────────────────────────┐
│  GitHub Repository (main branch)            │
│  └─ Triggers GitHub Pages build            │
└────────────────┬────────────────────────────┘
                 │
                 ↓ GitHub Actions
┌─────────────────────────────────────────────┐
│  Jekyll Build Process                       │
│  ├─ Read _config.yml                        │
│  ├─ Process .md files → .html              │
│  ├─ Apply Primer theme                      │
│  ├─ Generate site in _site/                │
│  └─ Upload to GitHub Pages                 │
└────────────────┬────────────────────────────┘
                 │
                 ↓ 2-5 minutes
┌─────────────────────────────────────────────┐
│  Live Website                                │
│  https://codebyshennan.github.io/           │
│         tamkeen-data/                        │
│                                              │
│  Students access enhanced materials! 🎉     │
└──────────────────────────────────────────────┘
```

---

## 🎓 For Students

Once deployed, students can:

1. **View in browser**: Visit the live URL
2. **No installation needed**: Everything runs in the browser
3. **Interactive notebooks**: Click Colab links to run code
4. **Watch videos**: Embedded YouTube videos
5. **Visualize code**: Python Tutor links work immediately

---

## 📝 Quick Reference

### Useful Commands

```bash
# Check current remote
git remote -v

# Check current branch
git branch

# See what will be deployed
git status

# Preview locally
cd docs && bundle exec jekyll serve

# Deploy
git push origin main

# Check deployment status
# Visit: https://github.com/codebyshennan/tamkeen-data/actions
```

### Important URLs

- **Repository**: https://github.com/codebyshennan/tamkeen-data
- **Live Site**: https://codebyshennan.github.io/tamkeen-data
- **Settings**: https://github.com/codebyshennan/tamkeen-data/settings/pages
- **Actions**: https://github.com/codebyshennan/tamkeen-data/actions

---

## 🚀 Ready to Deploy?

Your enhanced Python materials are ready! Just:

1. Uncomment the excluded paths in `_config.yml`
2. Push to GitHub
3. Students can access them at the live URL!

**Questions?** Check the GitHub Pages documentation: https://docs.github.com/en/pages
