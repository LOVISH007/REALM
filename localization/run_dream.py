import os
import sys
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from localization.compute_heatmaps import generate_heat_map, plot_and_save_heatmap

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config import BASE_DIR
except ImportError:
    # Fallback if config not found
    BASE_DIR = Path(__file__).parent.parent


def load_data(csv_path: str) -> pd.DataFrame:
    """Load image descriptions from CSV file."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} image records")
    return df

def process_single_image(df: pd.DataFrame, filename: str, images_dir: Path, output_dir: Path, 
                        window_size: int = 64, stride: int = 32):
    """Process a single image and generate its heatmap."""
    # Find the image in the dataframe
    image_row = df[df['filename'] == filename]
    
    if image_row.empty:
        print(f"Error: Image {filename} not found in the CSV file!")
        return False
    
    # Get image info
    description = image_row.iloc[0]['description']
    mos_score = image_row.iloc[0]['MOS']
    
    # Load image
    image_path = images_dir / filename
    if not image_path.exists():
        print(f"Error: Image file {image_path} not found!")
        return False
    
    print(f"\nProcessing image: {filename}")
    image = Image.open(image_path).convert("RGB")
    
    # Create output filename
    output_filename = f"heatmap_{filename}"
    output_path = output_dir / output_filename
    
    # Generate heatmap
    heatmap = plot_and_save_heatmap(
        image=image,
        text=description,
        save_path=str(output_path),
        window_size=window_size,
        stride=stride
    )
    
    return True

def process_multiple_images(df: pd.DataFrame, filenames: list, images_dir: Path, output_dir: Path,
                           window_size: int = 64, stride: int = 32):
    """Process multiple images and generate their heatmaps."""
    print(f"\nProcessing {len(filenames)} images...")
    
    for i, filename in enumerate(filenames, 1):
        print(f"\n{'='*50}")
        print(f"Processing {i}/{len(filenames)}: {filename}")
        print(f"{'='*50}")
        
        success = process_single_image(df, filename, images_dir, output_dir, window_size, stride)
        if not success:
            print(f"Failed to process {filename}")
        else:
            print(f"Successfully processed {filename}")

def main():
    """Main function to run heatmap generation."""
    print("Realness Project - Heatmap Generation")
    print("=" * 50)
    
    # Set up paths
    test_csv_path = BASE_DIR / "datasets" / "test" / "image_descriptions.csv"
    test_images_dir = BASE_DIR / "datasets" / "test" / "images"
    output_dir = BASE_DIR / "localization" / "heatmaps"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Check if paths exist
    if not test_csv_path.exists():
        print(f"Error: CSV file not found at {test_csv_path}")
        return
    
    if not test_images_dir.exists():
        print(f"Error: Images directory not found at {test_images_dir}")
        return
    
    # Load data
    df = load_data(test_csv_path)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            # Process all images
            filenames = df['filename'].tolist()
            print(f"Processing all {len(filenames)} images...")
        else:
            # Process specific images
            filenames = sys.argv[1:]
    else:
        # Default: process f22.png
        filenames = ["f22.png"]
    
    # Set processing parameters
    window_size = 64
    stride = 8
    
    if len(sys.argv) > 2 and "--window" in sys.argv:
        try:
            window_idx = sys.argv.index("--window")
            window_size = int(sys.argv[window_idx + 1])
        except (ValueError, IndexError):
            print("Warning: Invalid window size, using default 64")
    
    if len(sys.argv) > 2 and "--stride" in sys.argv:
        try:
            stride_idx = sys.argv.index("--stride")
            stride = int(sys.argv[stride_idx + 1])
        except (ValueError, IndexError):
            print("Warning: Invalid stride, using default 32")
    
    print(f"Using window_size={window_size}, stride={stride}")
    print(f"Output directory: {output_dir}")
    
    # Process images
    process_multiple_images(df, filenames, test_images_dir, output_dir, window_size, stride)
    
    print(f"\nHeatmap generation completed!")
    print(f"Check output directory: {output_dir}")

if __name__ == "__main__":
    main()
