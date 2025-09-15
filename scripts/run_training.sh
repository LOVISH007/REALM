#!/bin/bash

# Script to run the regression training
# Author: Lovish Kaushik
# Description: This script sets up the environment and runs the MOS prediction model training

set -e  

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "===================================================================================="
echo "Realness Project - Training Script"
echo "===================================================================================="
echo "Project root: $PROJECT_ROOT"

export PROJECT_ROOT="$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Handle virtual environment
VENV_DIR="$PROJECT_ROOT/venv"

if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "No virtual environment detected."
    
    if [[ -d "$VENV_DIR" ]]; then
        echo "Found existing virtual environment at $VENV_DIR"
        echo "Activating virtual environment..."
        source "$VENV_DIR/bin/activate"
    else
        echo "Creating new virtual environment at $VENV_DIR..."
        python3 -m venv "$VENV_DIR"
        echo "Activating virtual environment..."
        source "$VENV_DIR/bin/activate"
        echo "Upgrading pip..."
        pip install --upgrade pip
    fi
    echo ""
else
    echo "Using existing virtual environment: $VIRTUAL_ENV"
    echo ""
fi

# Install dependencies
if [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
    echo "Installing dependencies from pyproject.toml..."
    pip install -e . --quiet
    echo "Done"
    echo ""
elif [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r "$PROJECT_ROOT/requirements.txt" --quiet
    echo "Done"
    echo ""
else
    echo "Warning: Neither pyproject.toml nor requirements.txt found."
    echo "You may need to install dependencies manually."
    echo "Done"
    echo ""
fi

# Create output directory for models 
mkdir -p "$PROJECT_ROOT/regression/outputs"
mkdir -p "$PROJECT_ROOT/regression/models"

echo "===================================================================================="
echo "Starting Training..."
echo "===================================================================================="
echo "Training will save outputs to: $PROJECT_ROOT/regression/"
echo "Logs will be displayed below:"
echo ""

cd "$PROJECT_ROOT"

echo "Running: python3 -m regression.train"
echo ""

if python3 -m regression.train; then
    echo ""
    echo "===================================================================================="
    echo "Training completed successfully!"
    echo "===================================================================================="
    echo "Check the following locations for outputs:"
    echo "- Model: $PROJECT_ROOT/regression/outputs/best_model.pth"
    echo "- Training curves: $PROJECT_ROOT/regression/training_curves.png"
    echo ""
else
    echo ""
    echo "===================================================================================="
    echo "Training failed!"
    echo "===================================================================================="
    echo "Please check the error messages above."
    echo ""
    exit 1
fi

echo "Script completed successfully!"
