import os
import re
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO
import contextlib
import traceback
from pathlib import Path

# Create assets directory if it doesn't exist
ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(exist_ok=True)

# List of files to process
files = [
    "understanding-relationships.md",
    "correlation-analysis.md",
    "simple-linear-regression.md",
    "multiple-linear-regression.md",
    "model-diagnostics.md"
]

def extract_code_blocks(content):
    """Extract Python code blocks from markdown content."""
    # Match code blocks that start with ```python and end with ```
    pattern = r"```python\n(.*?)```"
    return re.findall(pattern, content, re.DOTALL)

@contextlib.contextmanager
def capture_output():
    """Capture stdout and stderr."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = mystdout = StringIO()
    sys.stderr = mystderr = StringIO()
    try:
        yield mystdout, mystderr
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def process_file(filename):
    """Process a single markdown file."""
    print(f"Processing {filename}...")
    
    # Read markdown content
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract code blocks
    code_blocks = extract_code_blocks(content)
    print(f"Found {len(code_blocks)} code blocks in {filename}")
    
    # Tracking the figures generated
    updated_content = content
    
    # Create a shared namespace for all code blocks in this file
    namespace = {
        "np": np,
        "pd": pd,
        "plt": plt,
        "__name__": "__main__"
    }
    
    # Add other common imports that might be needed
    exec("from scipy import stats", namespace)
    exec("import seaborn as sns", namespace)
    exec("from sklearn.linear_model import LinearRegression", namespace)
    exec("from sklearn.metrics import r2_score", namespace)
    exec("from sklearn.feature_selection import SelectKBest, f_regression, RFE", namespace)
    exec("from statsmodels.stats.stattools import durbin_watson", namespace)
    exec("from statsmodels.stats.outliers_influence import variance_inflation_factor", namespace)
    
    # Execute each code block
    for i, code in enumerate(code_blocks):
        if not code.strip():
            continue
            
        print(f"Executing code block {i+1}/{len(code_blocks)}...")
        
        # Clear any previous plots
        plt.close('all')
        
        # Store original code block
        original_code_block = f"```python\n{code}```"
        
        # Capture output and execute code
        try:
            with capture_output() as (stdout, stderr):
                # Execute code
                exec(code, namespace)
                
                # Prepare to collect outputs and images
                outputs_to_add = ""
                
                # Check if there's a plot to save
                if plt.get_fignums():
                    # Generate a unique filename for each plot
                    file_base = Path(filename).stem
                    fig_path = ASSETS_DIR / f"{file_base}_fig_{i+1}.png"
                    
                    # Ensure tight layout for all figures
                    for fig_num in plt.get_fignums():
                        fig = plt.figure(fig_num)
                        fig.tight_layout()
                    
                    # Save the figure
                    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    print(f"Saved figure to {fig_path}")
                    
                    # Create markdown image link
                    outputs_to_add += f"\n\n![{file_base}_fig_{i+1}]({ASSETS_DIR}/{fig_path.name})\n"
                
                # Get stdout if available
                output = stdout.getvalue()
                if output.strip():
                    print(f"Code block produced output: {output.strip()}")
                    # Don't include "Saved figure to..." messages in the markdown output
                    cleaned_output = "\n".join([line for line in output.strip().split("\n") 
                                              if not line.startswith("Saved figure to")])
                    if cleaned_output.strip():
                        outputs_to_add += f"\n```\n{cleaned_output}\n```\n"
                
                # Add all outputs after the code block
                if outputs_to_add:
                    updated_content = updated_content.replace(
                        original_code_block,
                        f"{original_code_block}{outputs_to_add}"
                    )
                    
        except Exception as e:
            print(f"Error executing code block {i+1}: {e}")
            print(traceback.format_exc())
    
    # Write updated content back to file
    if updated_content != content:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"Updated {filename} with generated outputs")
    else:
        print(f"No changes needed for {filename}")

def process_model_diagnostics():
    """Special processing for model-diagnostics.md which has dependent function definitions."""
    filename = "model-diagnostics.md"
    print(f"Processing {filename} with special handling...")
    
    # Read markdown content
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract code blocks
    code_blocks = extract_code_blocks(content)
    print(f"Found {len(code_blocks)} code blocks in {filename}")
    
    # Tracking the figures generated
    updated_content = content
    
    # Create a shared namespace for all code blocks in this file
    namespace = {
        "np": np,
        "pd": pd,
        "plt": plt,
        "__name__": "__main__"
    }
    
    # Add other common imports that might be needed
    exec("from scipy import stats", namespace)
    exec("import seaborn as sns", namespace)
    exec("from sklearn.linear_model import LinearRegression", namespace)
    
    # First pass: Define all functions
    for i, code in enumerate(code_blocks):
        if not code.strip():
            continue
        
        # If the code block defines a function, execute it to add to namespace
        if "def " in code:
            try:
                exec(code, namespace)
                print(f"Defined function from code block {i+1}")
            except Exception as e:
                print(f"Error defining function in code block {i+1}: {e}")
    
    # Second pass: Execute all code blocks and capture output/figures
    for i, code in enumerate(code_blocks):
        if not code.strip():
            continue
            
        print(f"Executing code block {i+1}/{len(code_blocks)}...")
        
        # Clear any previous plots
        plt.close('all')
        
        # Store original code block
        original_code_block = f"```python\n{code}```"
        
        # Capture output and execute code
        try:
            with capture_output() as (stdout, stderr):
                # Execute code
                exec(code, namespace)
                
                # Prepare to collect outputs and images
                outputs_to_add = ""
                
                # Check if there's a plot to save
                if plt.get_fignums():
                    # Generate a unique filename for each plot
                    file_base = Path(filename).stem
                    fig_path = ASSETS_DIR / f"{file_base}_fig_{i+1}.png"
                    
                    # Ensure tight layout for all figures
                    for fig_num in plt.get_fignums():
                        fig = plt.figure(fig_num)
                        try:
                            fig.tight_layout()
                        except:
                            # Some complex figures might have layout issues
                            pass
                    
                    # Save the figure
                    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    print(f"Saved figure to {fig_path}")
                    
                    # Create markdown image link
                    outputs_to_add += f"\n\n![{file_base}_fig_{i+1}]({ASSETS_DIR}/{fig_path.name})\n"
                
                # Get stdout if available
                output = stdout.getvalue()
                if output.strip():
                    print(f"Code block produced output: {output.strip()}")
                    # Don't include "Saved figure to..." messages in the markdown output
                    cleaned_output = "\n".join([line for line in output.strip().split("\n") 
                                              if not line.startswith("Saved figure to")])
                    if cleaned_output.strip():
                        outputs_to_add += f"\n```\n{cleaned_output}\n```\n"
                
                # Add all outputs after the code block
                if outputs_to_add:
                    updated_content = updated_content.replace(
                        original_code_block,
                        f"{original_code_block}{outputs_to_add}"
                    )
                    
        except Exception as e:
            print(f"Error executing code block {i+1}: {e}")
            print(traceback.format_exc())
    
    # Write updated content back to file
    if updated_content != content:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"Updated {filename} with generated outputs")
    else:
        print(f"No changes needed for {filename}")

def clean_markdown_file(filename):
    """Removes all image references and code outputs from the markdown file."""
    print(f"Cleaning previous outputs from {filename}...")
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove image links
    content = re.sub(r'!\[.*?\]\(assets/.*?\)\n', '', content)
    
    # Remove code outputs (anything between ```python...``` and the next section)
    content = re.sub(r'```python(.*?)```\n```(.*?)```\n', r'```python\1```\n', content, flags=re.DOTALL)
    
    # Remove any remaining output blocks after code blocks
    content = re.sub(r'```python(.*?)```\n```.*?```\n', r'```python\1```\n', content, flags=re.DOTALL)
    
    # Write the cleaned content back to the file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Cleaned {filename}")
    return content

def main():
    """Process all markdown files."""
    print("Starting code execution and output generation...")
    
    # First pass: clean all markdown files
    for file in files:
        clean_markdown_file(file)
    
    # Second pass: process all files
    for file in files:
        if file == "model-diagnostics.md":
            process_model_diagnostics()
        else:
            process_file(file)
    
    print("All files processed.")

if __name__ == "__main__":
    main()
