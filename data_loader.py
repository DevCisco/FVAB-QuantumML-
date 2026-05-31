import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from medmnist import OCTMNIST
import torchvision.transforms as transforms
import pandas as pd
import os

def get_data_loaders(seed, batch_size=128, resize=224, split_dir="dataset_splits"):

    data_transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    full_train = OCTMNIST(split='train', transform=data_transform, download=True, as_rgb=True)
    full_val   = OCTMNIST(split='val',   transform=data_transform, download=True, as_rgb=True)
    full_test  = OCTMNIST(split='test',  transform=data_transform, download=True, as_rgb=True)
    full_dataset = ConcatDataset([full_train, full_val, full_test])

    # Lettura degli ID per il seed richiesto
    try:
        train_ids = pd.read_csv(os.path.join(split_dir, f"train_ids_{seed}.csv"))["sample_index"].tolist()
        val_ids   = pd.read_csv(os.path.join(split_dir, f"val_ids_{seed}.csv"))["sample_index"].tolist()
        test_ids  = pd.read_csv(os.path.join(split_dir, f"test_ids_{seed}.csv"))["sample_index"].tolist()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File degli ID per seed={seed} non trovati! Esegui prima 'generate_splits.py'."
        )

    train_ds = Subset(full_dataset, train_ids)
    val_ds   = Subset(full_dataset, val_ids)
    test_ds  = Subset(full_dataset, test_ids)

    train_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,  generator=train_generator)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader