#!/bin/bash

echo "Installing required packages..."
python -m pip install -r requirements.txt

echo "Executing code blocks and generating outputs..."
python generate_outputs.py

echo "Done!"
