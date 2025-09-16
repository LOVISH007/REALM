#!/bin/bash

# Script to run model inference
# Author: Realness-Project Team
# Description: This script runs inference on the test dataset using the trained MOS prediction model
# Usage: ./run_inference.sh [--model MODEL_PATH] [image_filename1] [image_filename2] ... [--all]
# Examples:
#   ./run_inference.sh                                    # Use default model, run on full test set
#   ./run_inference.sh f22.png f126.png                   # Use default model, run on specific images
#   ./run_inference.sh --model /path/to/model.pth         # Use custom model, run on full test set
#   ./run_inference.sh --model /path/to/model.pth f22.png # Use custom model, run on specific images

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

# Parse command line arguments to check for custom model path
DEFAULT_MODEL_PATH="$PROJECT_ROOT/saved_models/best_model.pth"
CUSTOM_MODEL_PATH=""
IMAGE_ARGS=()

# Parse arguments for --model flag
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            CUSTOM_MODEL_PATH="$2"
            shift 2  # Remove --model and its value
            ;;
        *)
            IMAGE_ARGS+=("$1")  # Add to image arguments
            shift
            ;;
    esac
done

# Set the model path to use
if [[ -n "$CUSTOM_MODEL_PATH" ]]; then
    MODEL_PATH="$CUSTOM_MODEL_PATH"
    echo "Using custom model path: $MODEL_PATH"
else
    MODEL_PATH="$DEFAULT_MODEL_PATH"
    echo "Using default model path: $MODEL_PATH"
fi

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
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: Trained model not found at $MODEL_PATH"
    if [[ -n "$CUSTOM_MODEL_PATH" ]]; then
        echo "Please check the provided model path: $CUSTOM_MODEL_PATH"
    else
        echo "Please train the model first using: $PROJECT_ROOT/scripts/run_training.sh or put a model at $MODEL_PATH"
    fi
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

if [[ ${#IMAGE_ARGS[@]} -eq 0 ]]; then
    echo "Running inference on full test set (no specific images provided)"
else
    echo "Running inference on specific images: ${IMAGE_ARGS[*]}"
fi
echo ""

# Change to project directory
cd "$PROJECT_ROOT"

# Parse command line arguments for the inference script
if [[ ${#IMAGE_ARGS[@]} -eq 0 ]]; then
    echo "No image arguments provided. Running on full test set..."
    INFERENCE_ARGS=("--all")
else
    INFERENCE_ARGS=("${IMAGE_ARGS[@]}")
fi

# Add model path to inference arguments if custom model is provided
if [[ -n "$CUSTOM_MODEL_PATH" ]]; then
    INFERENCE_ARGS=("--model" "$MODEL_PATH" "${INFERENCE_ARGS[@]}")
fi

# Run the inference
echo "Running: python3 -m regression.inference ${INFERENCE_ARGS[*]}"
echo ""

if python3 -m regression.inference "${INFERENCE_ARGS[@]}"; then
    echo ""
    echo "=========================================="
    echo "Inference completed successfully!"
    echo "=========================================="
    echo "Check the following locations for outputs:"
    echo "- Predictions CSV: $PROJECT_ROOT/regression/test_predictions.csv"
    echo ""
fi

echo "Inference script completed successfully!"
