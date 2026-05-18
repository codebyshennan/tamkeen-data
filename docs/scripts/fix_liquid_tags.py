#!/usr/bin/env python3
"""
Script to remove unsupported Liquid tags from markdown files for Jekyll compatibility.
This fixes tags like {% stepper %}, {% step %}, {% tabs %}, etc.
"""

import re
import os
from pathlib import Path

# Patterns to remove (these tags are not supported by GitHub Pages Jekyll)
PATTERNS_TO_REMOVE = [
    # Stepper tags
    (r'\{%\s*stepper\s*%\}\s*\n?', ''),
    (r'\{%\s*endstepper\s*%\}\s*\n?', ''),
    # Step tags (replace with simple heading markers if preceded by content)
    (r'\{%\s*step\s*%\}\s*\n?', '\n---\n\n'),
    (r'\{%\s*endstep\s*%\}\s*\n?', '\n'),
    # Tabs (already fixed but including for completeness)
    (r'\{%\s*tabs\s*%\}\s*\n?', ''),
    (r'\{%\s*endtabs\s*%\}\s*\n?', ''),
    (r'\{%\s*tab\s+title="[^"]*"\s*%\}\s*\n?', '\n#### '),
    (r'\{%\s*endtab\s*%\}\s*\n?', '\n'),
    # Code tabs
    (r'\{%\s*code-tabs\s*%\}\s*\n?', ''),
    (r'\{%\s*endcode-tabs\s*%\}\s*\n?', ''),
    # Hints
    (r'\{%\s*hint\s+style="[^"]*"\s*%\}\s*\n?', '\n> **Note:** '),
    (r'\{%\s*endhint\s*%\}\s*\n?', '\n'),
]

def fix_liquid_tags(content):
    """Remove unsupported Liquid tags from content."""
    for pattern, replacement in PATTERNS_TO_REMOVE:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Clean up multiple consecutive newlines (max 2)
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content

def process_file(filepath):
    """Process a single markdown file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Check if file has unsupported tags
        has_unsupported_tags = any(
            re.search(pattern, original_content) 
            for pattern, _ in PATTERNS_TO_REMOVE
        )
        
        if not has_unsupported_tags:
            return False
        
        # Fix the tags
        fixed_content = fix_liquid_tags(original_content)
        
        # Only write if content changed
        if fixed_content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"✓ Fixed: {filepath}")
            return True
        
        return False
    except Exception as e:
        print(f"✗ Error processing {filepath}: {e}")
        return False

def main():
    """Find and fix all markdown files with unsupported Liquid tags."""
    docs_dir = Path(__file__).parent
    
    # Find all markdown files
    md_files = list(docs_dir.rglob('*.md'))
    
    print(f"Scanning {len(md_files)} markdown files...")
    
    fixed_count = 0
    for md_file in md_files:
        # Skip hidden directories and specific files
        if any(part.startswith('.') or part.startswith('_') for part in md_file.parts):
            continue
        
        if process_file(md_file):
            fixed_count += 1
    
    print(f"\n{'='*50}")
    print(f"Fixed {fixed_count} files")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
