import numpy as np
import os
from sklearn.model_selection import train_test_split

def get_stratified_fraction_indices(labels, fraction, seed):
    """
    Estrae una frazione del dataset mantenendo la proporzione delle classi (stratificato).
    """
    train_idx, _ = train_test_split(
        np.arange(len(labels)),
        train_size=fraction,
        stratify=labels,
        random_state=seed
    )
    return train_idx

def save_fewshot_manifest(indices, fraction, seed, d):
    """
    Salva gli indici per la riproducibilità (Freeze).
    """
    os.makedirs("artifacts/fewshot", exist_ok=True)
    filename = f"artifacts/fewshot/indices_f{int(fraction*100)}_s{seed}_d{d}.npy"
    np.save(filename, indices)
    return filename