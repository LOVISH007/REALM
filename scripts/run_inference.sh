#!/bin/bash

# Script to run model inference
# Author: Agnij Biswas
# Description: This script runs inference on the test dataset using the trained MOS prediction model

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "Realness Project - Model Inference"
echo "=========================================="
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

echo ""

# Check if trained model exists
MODEL_PATH="$PROJECT_ROOT/saved_models/best_model.pth"
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: Trained model not found at $MODEL_PATH"
    echo "Please train the model first using: $PROJECT_ROOT/scripts/run_training.sh or put a model at $MODEL_PATH"
    exit 1
else
    echo "✓ Found trained model: $MODEL_PATH"
fi

echo ""

# Install dependencies if needed
if [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
    echo "Installing/updating dependencies from pyproject.toml..."
    pip install -e . --quiet
    echo "Done"
    echo ""
elif [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
    echo "Installing/updating dependencies from requirements.txt..."
    pip install -r "$PROJECT_ROOT/requirements.txt"
    echo "Done"
    echo ""
fi

# Create output directories
mkdir -p "$PROJECT_ROOT/regression/outputs"

echo "=========================================="
echo "Starting Model Inference..."
echo "=========================================="
echo "Model path: $MODEL_PATH"
echo "Test data: $PROJECT_ROOT/datasets/test/"
echo "Output directory: $PROJECT_ROOT/regression/"
echo ""

if [[ $# -eq 0 ]]; then
    echo "Running inference on full test set (no specific images provided)"
else
    echo "Running inference on specific images: $@"
fi
echo ""

# Change to project directory
cd "$PROJECT_ROOT"

# Parse command line arguments for the inference script
if [[ $# -eq 0 ]]; then
    echo "No image arguments provided. Running on full test set..."
    INFERENCE_ARGS="--all"
else
    INFERENCE_ARGS="$@"
fi

# Run the inference
echo "Running: python3 -m regression.inference $INFERENCE_ARGS"
echo ""

if python3 -m regression.inference $INFERENCE_ARGS; then
    echo ""
    echo "=========================================="
    echo "Inference completed successfully!"
    echo "=========================================="
    echo "Check the following locations for outputs:"
    echo "- Predictions CSV: $PROJECT_ROOT/regression/test_predictions.csv"
    echo ""
fi

echo "Inference script completed successfully!"
