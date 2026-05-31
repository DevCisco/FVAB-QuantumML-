import os
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, random_split
from medmnist import OCTMNIST

SEEDS = [11, 17, 29]

def generate_and_save_ids(output_dir="dataset_splits"):
    """
    Genera e salva gli indici di train/val/test per ogni seed in CSV separati.
    Lo split viene eseguito in modo deterministico usando il seed come generatore.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Carichiamo solo per contare gli indici totali (nessuna transform pesante)
    full_train = OCTMNIST(split='train', download=True)
    full_val   = OCTMNIST(split='val',   download=True)
    full_test  = OCTMNIST(split='test',  download=True)

    dataset      = ConcatDataset([full_train, full_val, full_test])
    total_size   = len(dataset)
    train_size   = int(0.8 * total_size)
    val_size     = int(0.1 * total_size)
    test_size    = total_size - train_size - val_size

    print(f"Dataset totale: {total_size} campioni  "
          f"(train={train_size}, val={val_size}, test={test_size})")

    for seed in SEEDS:
        generator = torch.Generator().manual_seed(seed)
        train_idx, val_idx, test_idx = random_split(
            list(range(total_size)),
            [train_size, val_size, test_size],
            generator=generator
        )

        splits = {
            f"train_ids_{seed}.csv": list(train_idx),
            f"val_ids_{seed}.csv":   list(val_idx),
            f"test_ids_{seed}.csv":  list(test_idx),
        }

        for filename, indices in splits.items():
            out_path = os.path.join(output_dir, filename)
            pd.DataFrame(indices, columns=["sample_index"]).to_csv(out_path, index=False)
            print(f"  Salvato {filename} ({len(indices)} campioni)")

if __name__ == "__main__":
    generate_and_save_ids()