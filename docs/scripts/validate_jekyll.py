#!/usr/bin/env python3
"""
Validate markdown files for Jekyll/GitHub Pages compatibility.
Checks for unsupported Liquid tags that would cause build failures.
"""

import re
import sys
from pathlib import Path

# Unsupported Liquid tags patterns
UNSUPPORTED_PATTERNS = [
    (r'\{%\s*stepper\s*%\}', 'stepper'),
    (r'\{%\s*endstepper\s*%\}', 'endstepper'),
    (r'\{%\s*step\s*%\}', 'step'),
    (r'\{%\s*endstep\s*%\}', 'endstep'),
    (r'\{%\s*tabs\s*%\}', 'tabs'),
    (r'\{%\s*endtabs\s*%\}', 'endtabs'),
    (r'\{%\s*tab\s+', 'tab'),
    (r'\{%\s*endtab\s*%\}', 'endtab'),
    (r'\{%\s*code-tabs\s*%\}', 'code-tabs'),
    (r'\{%\s*endcode-tabs\s*%\}', 'endcode-tabs'),
    (r'\{%\s*hint\s+', 'hint'),
    (r'\{%\s*endhint\s*%\}', 'endhint'),
    (r'\{%\s*swagger\s+', 'swagger'),
    (r'\{%\s*embed\s+', 'embed'),
]

def check_file(filepath):
    """Check a single file for unsupported Liquid tags."""
    errors = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for pattern, tag_name in UNSUPPORTED_PATTERNS:
                if re.search(pattern, line):
                    errors.append({
                        'file': filepath,
                        'line': line_num,
                        'tag': tag_name,
                        'content': line.strip()
                    })
    except Exception as e:
        errors.append({
            'file': filepath,
            'line': 0,
            'tag': 'ERROR',
            'content': str(e)
        })
    
    return errors

def main(files=None):
    """
    Validate markdown files for Jekyll compatibility.
    
    Args:
        files: List of file paths to check. If None, checks all .md files in docs/
    """
    if files:
        # Check only specified files
        md_files = [Path(f) for f in files if f.endswith('.md')]
    else:
        # Check all markdown files in docs directory
        docs_dir = Path(__file__).parent
        md_files = [
            f for f in docs_dir.rglob('*.md')
            if not any(part.startswith('.') or part.startswith('_') for part in f.parts)
        ]
    
    if not md_files:
        print("No markdown files to check.")
        return 0
    
    print(f"Checking {len(md_files)} markdown files for Jekyll compatibility...")
    
    all_errors = []
    for md_file in md_files:
        errors = check_file(md_file)
        all_errors.extend(errors)
    
    if all_errors:
        print("\n" + "="*70)
        print("❌ VALIDATION FAILED - Unsupported Liquid tags found:")
        print("="*70)
        
        for error in all_errors:
            print(f"\n📄 File: {error['file']}")
            print(f"   Line {error['line']}: Unsupported tag '{{%  {error['tag']} %}}'")
            print(f"   Content: {error['content'][:100]}")
        
        print("\n" + "="*70)
        print("These tags are not supported by GitHub Pages Jekyll.")
        print("Run 'python3 docs/scripts/fix_liquid_tags.py' to automatically fix them.")
        print("="*70 + "\n")
        
        return 1
    
    print("✅ All files are Jekyll-compatible!")
    return 0

if __name__ == '__main__':
    # Get files from command line args (for pre-commit hook)
    files = sys.argv[1:] if len(sys.argv) > 1 else None
    sys.exit(main(files))
