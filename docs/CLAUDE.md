# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Data Science and AI educational course repository (Tamkeen program by Skills Union). It's a Jekyll-based static site with 6 progressive modules covering data fundamentals through machine learning, plus interactive Reveal.js slide presentations.

**Live site**: https://codebyshennan.github.io/tamkeen-data

## Build Commands

### Slide Generation
```bash
# Build all slides
make build

# Build specific modules
make prep                    # Module 0: Setup/preparation
make data-fundamentals       # Module 1: All submodules
make python                  # Module 1.2 only
make stats                   # Module 1.3 only
make linear-algebra          # Module 1.4 only
make pandas                  # Module 1.5 only

# Build any module directly
node slides/build.js <module-path>
# Example: node slides/build.js 2-data-wrangling/2.1-sql

# Clean generated files
make clean
make clean-module MODULE=<submodule-name>

# Regenerate specific module
make regenerate MODULE=<submodule-name>
```

### Jekyll (Local Development)
```bash
bundle exec jekyll serve      # Serve locally
bundle exec jekyll build      # Build static site
```

## Architecture

### Directory Structure
- `0-prep/` through `6-capstone/` - Course modules (numbered for progression)
- Each module contains:
  - Markdown content files (tutorials, concepts)
  - `slides/data.json` - Slide definitions for Reveal.js
  - `slides/index.html` - Generated presentations (do not edit directly)
  - `_assignments/` - Practical exercises

### Slide Generation System
1. Define slides in `<module>/slides/data.json` with type (`title` or `content`)
2. Run `node slides/build.js <module-path>`
3. Script reads `slides/template.html` and generates `slides/index.html`

### Content Patterns
- Module overviews in `README.md`
- Concept files follow kebab-case: `concept-name.md`
- MathJax enabled for mathematical notation (kramdown with mathjax engine)
- Python is the default syntax highlighting language

## Key Configuration

- **Package manager**: pnpm
- **Jekyll theme**: primer (remote theme)
- **Markdown**: kramdown with Rouge syntax highlighting
- **Presentations**: Reveal.js v4.3.1 (CDN-based)
