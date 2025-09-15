from scipy.stats import spearmanr,pearsonr
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer, BertModel
try:
    from regression.realism_dataset import RealismDataset
except ImportError:
    from .regression.realism_dataset import RealismDataset

from config import BASE_DIR

@torch.no_grad()
def evaluate_correlation_scores(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    for images, input_ids, attention_mask, mos in test_loader:
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        mos = mos.to(device)

        preds = model(images, input_ids, attention_mask)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(mos.cpu().numpy())

    corr, _ = spearmanr(all_targets, all_preds)
    pearsonr_corr, _ = pearsonr(all_targets, all_preds)
    return corr, pearsonr_corr


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    for images, input_ids, attention_mask, mos in val_loader:
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        mos = mos.to(device)

        preds = model(images, input_ids, attention_mask)
        loss = criterion(preds, mos)
        total_loss += loss.item()
    return total_loss / len(val_loader)


def create_data_loader(df, image_path, batch_size=16, split="train") -> DataLoader:
    """
    Create data loaders for training, validation, and testing

    Args:
        df (pd.DataFrame): DataFrame containing image filenames and MOS scores.
        image_path (Path): Path to the directory containing images.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(10),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    ])
    
    if split == "test":
        transform = transforms.Compose([
            transforms.Resize((384, 512)),
            transforms.ToTensor(),
        ])
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    dataset = RealismDataset(df, image_dir=image_path, tokenizer=tokenizer, transform=transform)
    
    # Create data loaders
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"))
    
    return dataloader


def plot_losses(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(BASE_DIR,filename)
        while os.path.exists(os.path.join(save_path, filename)):
            i += 1
            filename = f"loss_curve_{i}.png"

        file_save_path = os.path.join(BASE_DIR, filename)
        plt.savefig(file_save_path)
    else:
        plt.savefig(save_path)
