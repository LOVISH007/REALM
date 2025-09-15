import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
from scipy.stats import spearmanr, pearsonr
import os
import numpy as np
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

try:
    from config import BASE_DIR
    from regression.regression_model import MOSPredictor
    from utils import evaluate_correlation_scores, create_data_loader
except ImportError:
    from ..config import BASE_DIR
    from regression_model import MOSPredictor
    from ..utils import evaluate_correlation_scores, create_data_loader

# Setting seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
np.random.seed(seed)

TEST_IMG_PATH = BASE_DIR / "datasets" / "test" / "images"
TEST_CSV_PATH = BASE_DIR / "datasets" / "test" / "image_descriptions.csv"
MODEL_SAVE_PATH = BASE_DIR / "saved_models" / "best_model.pth"

def test_best_model(test_loader):
    """Load and test the best saved model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading best model for final evaluation...")
    model = MOSPredictor().to(device)
    
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print(f"Loaded model from: {MODEL_SAVE_PATH}")
    else:
        print("No saved model found. Using current model state.")
        
    spearman_corr, pearson_corr = evaluate_correlation_scores(model, test_loader, device)
    
    return spearman_corr, pearson_corr


def load_test_data():
    """Load and prepare the test dataset"""
    print("Loading test dataset...")
    
    # Load the test CSV file
    test_df = pd.read_csv(TEST_CSV_PATH)
    print(f"Loaded {len(test_df)} test samples")
    
    # Display basic statistics
    print(f"Test MOS range: {test_df['MOS'].min():.4f} to {test_df['MOS'].max():.4f}")
    print(f"Test MOS mean: {test_df['MOS'].mean():.4f}, std: {test_df['MOS'].std():.4f}")
    
    return test_df


def predict_single_image(model, image_path, description, device):
    """
    Predict MOS for a single image
    
    Args:
        model: Trained MOSPredictor model
        image_path: Path to the image file
        description: Text description of the image
        device: Device to run inference on
    
    Returns:
        float: Predicted MOS score
    """
    model.eval()
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.ToTensor(),
    ])
    
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    
    # Tokenize text
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(description, padding='max_length', truncation=True, 
                      return_tensors="pt", max_length=128)
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(image_tensor, input_ids, attention_mask)
        
    return prediction.cpu().item()


def run_inference_on_images(model, test_df, device, image_filenames):
    """
    Run inference on specific image filenames
    
    Args:
        model (nn.Module): trained model
        test_df (pd.DataFrame): test dataframe with all image data
        device (torch.device): device to run on
        image_filenames (list): list of image filenames to process
    """
    print(f"\nRunning inference on {len(image_filenames)} specific images...")
    print("=" * 60)
    
    results = []
    
    for filename in image_filenames:
        # Find image data in dataframe
        image_data = test_df[test_df['filename'] == filename]
        
        if image_data.empty:
            print(f"Warning: Image {filename} not found in dataset. Skipping...")
            continue
            
        image_row = image_data.iloc[0]
        image_path = TEST_IMG_PATH / filename
        
        if not image_path.exists():
            print(f"Warning: Image file {image_path} not found. Skipping...")
            continue
        
        description = image_row['description']
        true_mos = image_row['MOS']
        
        # Make prediction
        predicted_mos = predict_single_image(model, image_path, description, device)
        
        if predicted_mos is not None:            
            print(f"\nImage: {filename}")
            print(f"Description: {description}")
            print(f"True MOS: {true_mos:.4f}")
            print(f"Predicted MOS: {predicted_mos:.4f}")
            print("-" * 40)
            
            results.append({
                'filename': filename,
                'true_MOS': true_mos,
                'predicted_MOS': predicted_mos,
                'description': description
            })
        else:
            print(f"Error: Failed to process {filename}")
    
    if results:
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        return results_df
    else:
        print("No images were successfully processed.")
        return None


def main():
    print("MOS Prediction Model Inference")
    print("=" * 60)
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        run_full_test = True
        target_images = []
    else:
        if sys.argv[1] == "--all":
            run_full_test = True
            target_images = []
        else:
            run_full_test = False
            target_images = sys.argv[1:]
    
    # Check if required paths exist
    required_paths = [TEST_IMG_PATH, TEST_CSV_PATH, MODEL_SAVE_PATH]
    for path in required_paths:
        if not os.path.exists(path):
            print(f"Error: Required path not found: {path}")
            return
    
    # Load test data
    test_df = load_test_data()
    
    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MOSPredictor().to(device)

    results_save_path = BASE_DIR / "regression" / "outputs" / "test_predictions.csv"
    
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(f"Loaded model from: {MODEL_SAVE_PATH}")
    else:
        print("Error: No saved model found!")
        return
    
    if run_full_test:
        # Run inference on full test set
        print("\nRunning inference on full test set...")
        test_loader = create_data_loader(test_df, TEST_IMG_PATH, batch_size=16, split="test")
        spearman_corr, pearson_corr = test_best_model(test_loader)
        
        print("=" * 60)
        print(f"Total samples: {len(test_df)}")
        print(f"Spearman Correlation: {spearman_corr:.4f}")
        print(f"Pearson Correlation: {pearson_corr:.4f}")
        
    else:
        # Run inference on specific images
        results_df = run_inference_on_images(model, test_df, device, target_images)
        if results_df is not None:
            results_df.to_csv(results_save_path, index=False)
        print(f"Detailed results saved to: {results_save_path}")
    
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()

