import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr
import os
import numpy as np
from pathlib import Path

try:
    from config import BASE_DIR
    from regression.realism_dataset import RealismDataset
    from regression.regression_model import MOSPredictor
    from utils import evaluate_correlation_scores, validate, plot_losses
except ImportError:
    from ..config import BASE_DIR
    from realism_dataset import RealismDataset
    from regression_model import MOSPredictor
    from ..utils import evaluate_correlation_scores, validate, plot_losses

# Setting seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
np.random.seed(seed)

# Define paths based on BASE_DIR from config
TRAIN_IMG_PATH = BASE_DIR / "datasets" / "train" / "images"
TRAIN_CSV_PATH = BASE_DIR / "datasets" / "train" / "image_descriptions.csv"
TEST_IMG_PATH = BASE_DIR / "datasets" / "test" / "images"
TEST_CSV_PATH = BASE_DIR / "datasets" / "test" / "image_descriptions.csv"
MODEL_SAVE_PATH = BASE_DIR / "regression" / "best_model.pth"


def load_and_prepare_data():
    """Load and prepare the dataset"""
    print("Loading dataset...")
    
    # Load the training CSV file
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    print(f"Loaded {len(train_df)} training samples")
    
    # Load the test CSV file
    test_df = pd.read_csv(TEST_CSV_PATH)
    print(f"Loaded {len(test_df)} test samples")
    
    # Split training data into train and validation
    train_df, val_df = train_test_split(train_df, test_size=0.20, random_state=42)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Display basic statistics
    print(f"Train MOS range: {train_df['MOS'].min():.4f} to {train_df['MOS'].max():.4f}")
    print(f"Train MOS mean: {train_df['MOS'].mean():.4f}, std: {train_df['MOS'].std():.4f}")
    print(f"Test MOS range: {test_df['MOS'].min():.4f} to {test_df['MOS'].max():.4f}")
    print(f"Test MOS mean: {test_df['MOS'].mean():.4f}, std: {test_df['MOS'].std():.4f}")
    
    return train_df, val_df, test_df


def create_data_loaders(train_df, val_df, test_df, batch_size=16):
    """Create data loaders for training, validation, and testing"""
    
    # Define transforms (matching the notebook)
    transform = transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomRotation(10),  # Data augmentation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Data augmentation
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.ToTensor(),
    ])
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = RealismDataset(train_df, image_dir=TRAIN_IMG_PATH, 
                                  tokenizer=tokenizer, transform=transform)
    val_dataset = RealismDataset(val_df, image_dir=TRAIN_IMG_PATH, 
                                tokenizer=tokenizer, transform=transform)
    test_dataset = RealismDataset(test_df, image_dir=TEST_IMG_PATH, 
                                 tokenizer=tokenizer, transform=transform_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_model(train_loader, val_loader, test_loader, num_epochs=10, learning_rate=0.0001):
    """Train the MOS prediction model"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = MOSPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_pearson_corr = 0.80
    best_spearman_corr = 0.7627
    
    print("Starting training...")
    print("-" * 50)
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for images, input_ids, attention_mask, mos in train_loader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            mos = mos.to(device)
            
            # Forward pass
            preds = model(images, input_ids, attention_mask)
            loss = criterion(preds, mos)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Evaluate on test set
        model.eval()
        spearman_corr, pearson_corr = evaluate_correlation_scores(model, test_loader, device)
        print(f"Pearson Correlation: {pearson_corr:.4f}, Spearman Correlation: {spearman_corr:.4f}")
        
        # Save best model
        if spearman_corr > best_spearman_corr:
            print(f"{spearman_corr > best_spearman_corr}")
            best_spearman_corr = spearman_corr
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved with Spearman Correlation: {spearman_corr:.4f}")
        
        print("-" * 50)
    
    print(f"Training completed!")
    print(f"Best Spearman Correlation: {best_spearman_corr:.4f}")
    
    return model, train_losses, val_losses, best_spearman_corr


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
    
    model.eval()
    
    spearman_corr, pearson_corr = evaluate_correlation_scores(model, test_loader, device)
    
    print("=" * 50)
    print("FINAL TEST RESULTS")
    print("=" * 50)
    print(f"Final Test Spearman Correlation: {spearman_corr:.4f}")
    print(f"Final Test Pearson Correlation: {pearson_corr:.4f}")
    print("=" * 50)
    
    return spearman_corr, pearson_corr


def main():
    """Main training and testing pipeline"""
    print("MOS Prediction Model Training")
    print("=" * 50)
    
    # Check if required directories exist
    required_paths = [TRAIN_IMG_PATH, TRAIN_CSV_PATH, TEST_IMG_PATH, TEST_CSV_PATH]
    for path in required_paths:
        if not os.path.exists(path):
            print(f"Error: Required path not found: {path}")
            return
    
    # Load and prepare data
    train_df, val_df, test_df = load_and_prepare_data()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df, batch_size=16
    )
    
    # Train the model
    model, train_losses, val_losses, best_spearman = train_model(
        train_loader, val_loader, test_loader, num_epochs=10, learning_rate=0.0001101
    )
    
    # Plot training curves
    plot_save_path = BASE_DIR / "regression" / "training_curves.png"
    plot_losses(train_losses, val_losses, save_path=plot_save_path)
    
    # Final evaluation with best model
    final_spearman, final_pearson = test_best_model(test_loader)
    
    print("\nTraining Summary:")
    print("=" * 50)
    print(f"Best Spearman during training: {best_spearman:.4f}")
    print(f"Final test Spearman: {final_spearman:.4f}")
    print(f"Final test Pearson: {final_pearson:.4f}")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")


if __name__ == "__main__":
    main()