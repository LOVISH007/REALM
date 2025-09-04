import torch
from torch.utils.data import Dataset
import os
from PIL import Image

class RealismDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_dir, tokenizer, transform):
        self.df = df
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = f"{self.image_dir}/{row['filename']}"
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        text = row['description']
        inputs = self.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length=128)

        
        mos = torch.tensor(row['MOS'], dtype=torch.float)

        return image, inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), mos
