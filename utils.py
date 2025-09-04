from scipy.stats import spearmanr,pearsonr
import torch
import matplotlib.pyplot as plt
import os

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

    spearmanr_corr, _ = spearmanr(all_targets, all_preds)
    pearsonr_corr, _ = pearsonr(all_targets, all_preds)
    return spearmanr_corr, pearsonr_corr


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


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    filename = "loss_curve_1.png"
    while os.path.exists(os.path.join(BASE_DIR, filename)):
        i += 1
        filename = f"loss_curve_{i}.png"

    file_save_path = os.path.join(BASE_DIR, filename)
    plt.savefig(file_save_path)



