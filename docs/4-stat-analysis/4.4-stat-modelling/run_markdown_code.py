#!/usr/bin/env python3
"""
This utility script extracts and runs Python code blocks from markdown files,
saving the outputs and generated visualizations.

Usage:
    python run_markdown_code.py <markdown_file.md>

Example:
    python run_markdown_code.py logistic-regression.md

This will:
1. Extract all Python code blocks from the markdown file
2. Run them in order
3. Save any generated plots to the assets directory
4. Create an output log with the results

Requirements:
    pip install matplotlib numpy pandas scikit-learn seaborn
"""

import sys
import os
import re
from io import StringIO
import contextlib

# Import common data science libraries - these will be available to code blocks
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn import datasets, linear_model, metrics, preprocessing, model_selection
    import seaborn as sns
except ImportError as e:
    print(f"Warning: {e}")
    print("Some required packages may be missing. Install them with:")
    print("pip install matplotlib numpy pandas scikit-learn seaborn")


def extract_code_blocks(markdown_file):
    """Extract Python code blocks from a markdown file."""
    with open(markdown_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regular expression to match Python code blocks
    # Matches ```python or ``` python followed by code and ending with ```
    pattern = r'```(?:python|)\s*([\s\S]*?)```'
    
    # Find all code blocks
    code_blocks = re.findall(pattern, content)
    
    return code_blocks


def run_code_block(code_block, block_num, output_file):
    """Run a single code block and capture its output."""
    # Create a string buffer to capture stdout
    stdout_buffer = StringIO()
    
    # Ensure assets directory exists
    os.makedirs('assets', exist_ok=True)
    
    # Write the code block to the output file
    output_file.write(f"\n\n### Code Block {block_num}\n")
    output_file.write("```python\n")
    output_file.write(code_block)
    output_file.write("\n```\n\n")
    
    # Record any figures that existed before running the code
    pre_figures = plt.get_fignums()
    
    # Run the code block and capture standard output
    with contextlib.redirect_stdout(stdout_buffer):
        try:
            # Use exec to run the code
            exec(code_block, globals())
            success = True
        except Exception as e:
            print(f"Error: {str(e)}")
            success = False
    
    # Get the output
    output = stdout_buffer.getvalue()
    
    # Write the output to the file
    if output.strip():
        output_file.write("Output:\n")
        output_file.write("```\n")
        output_file.write(output)
        output_file.write("\n```\n\n")
    
    # Save any new figures that were generated
    post_figures = plt.get_fignums()
    new_figures = [fig for fig in post_figures if fig not in pre_figures]
    
    for fig_num in new_figures:
        fig = plt.figure(fig_num)
        filename = f"assets/block_{block_num}_figure_{fig_num}.png"
        fig.savefig(filename)
        output_file.write(f"![Figure {fig_num}]({filename})\n\n")
    
    # Close all new figures
    for fig_num in new_figures:
        plt.close(fig_num)
    
    return success


def main():
    """Main function to extract and run code blocks."""
    if len(sys.argv) < 2:
        print("Usage: python run_markdown_code.py <markdown_file.md>")
        return
    
    markdown_file = sys.argv[1]
    
    if not os.path.exists(markdown_file):
        print(f"Error: File '{markdown_file}' not found.")
        return
    
    # Extract code blocks
    code_blocks = extract_code_blocks(markdown_file)
    
    if not code_blocks:
        print(f"No Python code blocks found in {markdown_file}")
        return
    
    print(f"Found {len(code_blocks)} code blocks in {markdown_file}")
    
    # Create output file
    output_filename = f"{os.path.splitext(markdown_file)[0]}_output.md"
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write(f"# Code Execution Output: {markdown_file}\n\n")
        output_file.write(f"This file contains the output from running the code blocks in `{markdown_file}`.\n")
        
        # Run each code block
        for i, block in enumerate(code_blocks, 1):
            print(f"Running code block {i}...")
            success = run_code_block(block, i, output_file)
            status = "Succeeded" if success else "Failed"
            print(f"Code block {i}: {status}")
    
    print(f"Execution complete. Output saved to {output_filename}")


if __name__ == "__main__":
    main()
