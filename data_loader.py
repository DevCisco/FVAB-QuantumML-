import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from medmnist import OCTMNIST
import torchvision.transforms as transforms
import pandas as pd
import os

# Statistiche di normalizzazione usate in tutto il progetto per le immagini
# in ingresso al backbone ResNet18. Esposte come costanti — non solo come
# valori inline — così altri file (es. week8_robustness.py) possono costruire
# trasformazioni custom (es. con rumore) restando coerenti con quelle usate
# qui, invece di duplicare i valori con rischio di disallineamento silenzioso.
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD  = [0.5, 0.5, 0.5]


def get_data_loaders(seed, batch_size=128, resize=224, split_dir="dataset_splits",
                      transform=None):
    """
    Restituisce (train_loader, val_loader, test_loader) per una coppia (d, seed).

    Protocollo sperimentale corretto
    ---------------------------------
    Il test set è FISSO e identico per tutti i seed (letto da test_ids_fixed.csv).
    Il seed controlla esclusivamente la suddivisione train/val del pool di training.

    Composizione degli split (generata da generate_fixed_splits.py):
        test  (fisso):  canonical val + canonical test MedMNIST  → 11.832 immagini
                        Il backbone ResNet (pesi MedMNIST) NON ha mai visto
                        queste immagini: è addestrato solo sul canonical train.
        train (seed):   ~87.729 immagini dal canonical train pool
        val   (seed):   ~9.748  immagini dal canonical train pool

    Perché questo protocollo è corretto
    ------------------------------------
    1. Test fisso → la varianza inter-seed riflette solo la stabilità del modello
       (quale subset di training viene selezionato), non la difficoltà del test set.
       Con protocollo corretto, la varianza attesa è ~2-5% macro-F1, non ~11%.

    2. Nessun data leakage dal backbone → i pesi MedMNIST vengono da un ResNet
       addestrato sul canonical OCTMNIST train split. Il test fisso (canonical val
       + canonical test) non è mai entrato nei gradienti del backbone.

    Args:
        transform: trasformazione torchvision custom (opzionale). Se None
            (default), usa Resize+ToTensor+Normalize(NORM_MEAN, NORM_STD) —
            il comportamento standard del progetto. Permette di iniettare
            trasformazioni aggiuntive (es. rumore gaussiano) mantenendo lo
            stesso identico split fisso, necessario per confrontare in modo
            equo risultati "clean" e "noisy" sullo stesso test set
            (usato da week8_robustness.py).
    """

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])

    full_train   = OCTMNIST(split='train', transform=transform, download=True, as_rgb=True)
    full_val     = OCTMNIST(split='val',   transform=transform, download=True, as_rgb=True)
    full_test    = OCTMNIST(split='test',  transform=transform, download=True, as_rgb=True)
    full_dataset = ConcatDataset([full_train, full_val, full_test])

    try:
        train_ids = pd.read_csv(
            os.path.join(split_dir, f"train_ids_{seed}.csv")
        )["sample_index"].tolist()

        val_ids = pd.read_csv(
            os.path.join(split_dir, f"val_ids_{seed}.csv")
        )["sample_index"].tolist()

        test_ids = pd.read_csv(
            os.path.join(split_dir, "test_ids_fixed.csv")
        )["sample_index"].tolist()

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\n"
            "Esegui prima 'generate_fixed_splits.py' per generare i CSV corretti.\n"
            "I vecchi CSV (train_ids_{seed}.csv, val_ids_{seed}.csv, test_ids_{seed}.csv) "
            "non sono compatibili con questo protocollo e vanno rigenerati."
        )

    train_ds = Subset(full_dataset, train_ids)
    val_ds   = Subset(full_dataset, val_ids)
    test_ds  = Subset(full_dataset, test_ids)

    train_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              generator=train_generator)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader