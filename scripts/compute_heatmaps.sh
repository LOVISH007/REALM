#!/bin/bash

# Script to run heatmap analysis
# Author: Agnij Biswas 
# Description: This script runs the heatmap generation for image-text similarity analysis

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "===================================================================================="
echo "Realness Project - Heatmap Analysis"
echo "===================================================================================="
echo "Project root: $PROJECT_ROOT"

# Set environment variables
export PROJECT_ROOT="$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

echo "Python version: $(python3 --version)"
echo ""

# Handle virtual environment
VENV_DIR="$PROJECT_ROOT/venv"

echo $PROJECT_ROOT

if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "No virtual environment detected."
    
    if [[ -d "$VENV_DIR" ]]; then
        echo "Found existing virtual environment at $VENV_DIR"
        echo "Activating virtual environment..."
        source "$VENV_DIR/bin/activate"
    else
        echo "Warning: No virtual environment found."
        echo "It's recommended to run this in a virtual environment."
        echo "To create one, run: python3 -m venv $VENV_DIR"
        echo ""
    fi
else
    echo "Using existing virtual environment: $VIRTUAL_ENV"
    echo ""
fi


# Install dependencies if needed
if [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
    echo "Installing/updating dependencies from pyproject.toml..."
    pip install -e . --quiet
    echo "Done"
    echo ""
elif [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
    echo "Installing/updating dependencies from requirements.txt..."
    pip install -r "$PROJECT_ROOT/requirements.txt" --quiet
    echo "Done"
    echo ""
fi

# Create output directories
mkdir -p "$PROJECT_ROOT/localization/heatmaps"

# Change to project directory
cd "$PROJECT_ROOT"

# Parse command line arguments
if [[ $# -eq 0 ]]; then
    # Default: run with f22.png
    echo "No arguments provided. Running with default image: f22.png"
    echo "Usage: $0 [image_filename(s)] [--all] [--window SIZE] [--stride SIZE]"
    echo ""
    echo "Examples:"
    echo "  $0 f22.png                    # Process single image"
    echo "  $0 f22.png f126.png          # Process multiple images"
    echo "  $0 --all                     # Process all images"
    echo "  $0 f22.png --window 128      # Use custom window size"
    echo "  $0 f22.png --stride 64       # Use custom stride"
    echo ""
    
    args=("f22.png")
else
    args=("$@")
fi

echo "===================================================================================="
echo "Starting Heatmap Analysis..."
echo "===================================================================================="
echo "Arguments: ${args[*]}"
echo ""

# Run the heatmap analysis
echo "Running: python3 localization/run_dream.py ${args[*]}"
echo ""

if python3 localization/run_dream.py "${args[@]}"; then
    echo ""
    echo "===================================================================================="
    echo "Heatmap Analysis completed successfully!"
    echo "===================================================================================="
    echo ""

# fid files:"
    ls -la "$PROJECT_ROOT/localization/heatmaps/" | grep -E "\\.png$" | head -10
    if [[ $(ls "$PROJECT_ROOT/localization/heatmaps/"*.png 2>/dev/null | wc -l) -gt 10 ]]; then
        echo "... and more files"
    fi
    echo ""
else
    echo ""
    echo "===================================================================================="
    echo "Heatmap Analysis failed!"
    echo "===================================================================================="
    echo "Please check the error messages above."
    exit 1
fi

echo "Script completed successfully!"
