import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from src.transforms import get_data_transforms


class SpectrogramDataset(Dataset):
    def __init__(self, spectrogram_folder, label_file_path, label_map, transform=None):
        self.spectrogram_folder = spectrogram_folder
        self.label_map = label_map

        # Use default transform if none provided
        self.transform = transform if transform is not None else get_data_transforms()

        # Load labels
        label_df = pd.read_csv(label_file_path, header=None, names=["filename", "label"])
        label_df["filename"] = label_df["filename"].apply(lambda x: os.path.splitext(x)[0])
        self.labels_dict = dict(zip(label_df["filename"], label_df["label"]))

        # List PNG files
        self.files = [f for f in os.listdir(spectrogram_folder) if f.endswith(".png")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        file_basename = os.path.splitext(filename)[0]
        image_path = os.path.join(self.spectrogram_folder, filename)
        image = Image.open(image_path).convert("RGB")

        # Apply transform
        image = self.transform(image)

        # Get label
        label_name = self.labels_dict.get(file_basename)
        label = self.label_map.get(label_name, -1)
        if label == -1:
            raise ValueError(f"Label for {filename} not found in label_map.")

        return image, label


# Dataloader function

def create_dataloaders(dataset, batch_size=4, train_ratio=0.8, seed=42):
    """
    Splits dataset into train/val/test and returns DataLoaders.
    Uses the same transform for all splits (no augmentation).
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    temp_size = total_size - train_size

    generator = torch.Generator().manual_seed(seed)

    # Split train and temp
    train_dataset, temp_dataset = random_split(dataset, [train_size, temp_size], generator=generator)

    # Split temp into val and test
    val_size = test_size = temp_size // 2
    val_dataset, test_dataset = random_split(temp_dataset, [val_size, temp_size - val_size], generator=generator)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader