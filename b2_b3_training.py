import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from unsupervised_models import VanillaAE, RegularizedAE


# ---------------------------------------------------------------------------
# Costanti globali
# ---------------------------------------------------------------------------
DIMS           = [32, 16, 8, 4]
SEEDS          = [11, 17, 29]
BACKBONE       = 'resnet'
EPOCHS         = 30
BATCH_SIZE     = 256
LR             = 1e-3
RECONSTRUCTION = nn.MSELoss()

# ---------------------------------------------------------------------------
# Caricamento features
# ---------------------------------------------------------------------------
def load_features(split: str, d: int, seed: int) -> torch.Tensor:
    """
    Carica le feature ResNet18 raw (512 dim) dal file .npz prodotto da test.py.

    Path NPZ: artifacts/resnet/features/res_raw_d_{d}_{split}_s{seed}.npz

    Questi file vengono generati da test.py nella fase 2 (save_raw_features),
    PRIMA della riduzione PCA. Contengono sempre feature a 512 dim
    indipendentemente da d — d è il bottleneck interno dell'AE, non
    la dimensione dell'input.

    Flusso completo del progetto:
        Immagine -> ResNet18 -> 512 dim -> PCA → d dim → AE (B2/B3) e VQC/few-shot

    Args:
        split (str): 'train' | 'val'.
        d (int): dimensione del bottleneck latente.
        seed (int): seed per riproducibilità.

    Returns:
        Tensor shape (N, 512), dtype float32.
    """
    
    # Path NPZ
    npz_path = f"artifacts/resnet/features/res_raw_d_{d}_{split}_s{seed}.npz"
    
    # Leggi NPZ direttamente in memoria
    data = np.load(npz_path)
    features = data['features'].astype(np.float32)
    
    return torch.tensor(features, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training B2: VanillaAE
# ---------------------------------------------------------------------------
def train_vanilla_ae(d: int, seed: int) -> dict:
    """
    Addestra VanillaAE (B2) per una coppia (d, seed).

    Architettura: 512 → 128 → d → 128 → 512
    Artifact salvati in: artifacts/sweep/B2_pca_{split}_d{d}_seed{seed}.csv
    Path atteso da week8_robustness.py.

    Args:
        d    (int): dimensione del bottleneck latente (4/8/16/32).
        seed (int): seed per riproducibilità del batch shuffle.

    Returns:
        dict con chiavi: model, d, seed, best_val_loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = load_features('train', d, seed).to(device)
    X_val   = load_features('val', d, seed).to(device)

    model     = VanillaAE(d_latent=d).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    rng = torch.Generator()
    rng.manual_seed(seed)

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):

        # — Train ------------------------------------------------------------
        model.train()
        idx   = torch.randperm(len(X_train), generator=rng)[:BATCH_SIZE]
        batch = X_train[idx]

        optimizer.zero_grad()
        recon, _ = model(batch)
        loss = RECONSTRUCTION(recon, batch)
        loss.backward()
        optimizer.step()

        # — Validation -------------------------------------------------------
        model.eval()
        with torch.no_grad():
            idx_val   = torch.randperm(len(X_val), generator=rng)[:BATCH_SIZE]
            val_batch = X_val[idx_val]
            val_recon, _ = model(val_batch)
            val_loss = RECONSTRUCTION(val_recon, val_batch).item()

        # — Checkpoint -------------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  [B2] d={d:2d} | seed={seed} | "
                f"Epoch {epoch+1:3d}/{EPOCHS} | "
                f"Train Loss: {loss.item():.6f} | Val Loss: {val_loss:.6f}",
                flush=True,
            )

    # — Salva latent features in CSV ----------------------------------------------
    model.eval()
    with torch.no_grad():
        for split, X in [('train', X_train), ('val', X_val)]:
            _, latent = model(X)
            latent_np = latent.cpu().numpy()
            df = pd.DataFrame(latent_np, columns=[f"latent_{i}" for i in range(d)])
            csv_path = f"artifacts/sweep/B2_pca_{split}_d{d}_seed{seed}.csv"
            df.to_csv(csv_path, index=False)

    print(f"  [B2] d={d:2d} | seed={seed} → Best Val Loss: {best_val_loss:.6f} | Latent features salvate")
    return {"model": "B2", "d": d, "seed": seed, "best_val_loss": best_val_loss}


# ---------------------------------------------------------------------------
# Training B3: RegularizedAE
# ---------------------------------------------------------------------------
def train_regularized_ae(d: int, seed: int) -> dict:
    """
    Addestra RegularizedAE (B3) per una coppia (d, seed).

    Architettura: 512 → d (shallow, sigmoid) → 512
    Loss totale = MSE(recon, x) + model.sparsity_loss(z)
    Val loss = solo MSE per confronto equo con B2.

    Artifact salvati in: artifacts/sweep/B3_pca_{split}_d{d}_seed{seed}.csv
    Path atteso da week8_robustness.py.

    Args:
        d    (int): dimensione del bottleneck latente (4/8/16/32).
        seed (int): seed per riproducibilità del batch shuffle.

    Returns:
        dict con chiavi: model, d, seed, best_val_loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = load_features('train', d, seed).to(device)
    X_val   = load_features('val', d, seed).to(device)

    model     = RegularizedAE(d_latent=d).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    rng = torch.Generator()
    rng.manual_seed(seed)

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):

        # — Train: MSE + L1 sparsità ----------------------------------------
        model.train()
        idx   = torch.randperm(len(X_train), generator=rng)[:BATCH_SIZE]
        batch = X_train[idx]

        optimizer.zero_grad()
        recon, z = model(batch)
        loss = RECONSTRUCTION(recon, batch) + model.sparsity_loss(z)
        loss.backward()
        optimizer.step()

        # — Validation: solo MSE --------------------------------------------
        model.eval()
        with torch.no_grad():
            idx_val   = torch.randperm(len(X_val), generator=rng)[:BATCH_SIZE]
            val_batch = X_val[idx_val]
            val_recon, _ = model(val_batch)
            val_loss = RECONSTRUCTION(val_recon, val_batch).item()

        # — Checkpoint -------------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  [B3] d={d:2d} | seed={seed} | "
                f"Epoch {epoch+1:3d}/{EPOCHS} | "
                f"Train Loss: {loss.item():.6f} | Val Loss: {val_loss:.6f}",
                flush=True,
            )

    # — Salva latent features in CSV ----------------------------------------------
    model.eval()
    with torch.no_grad():
        for split, X in [('train', X_train), ('val', X_val)]:
            _, latent = model(X)
            latent_np = latent.cpu().numpy()
            df = pd.DataFrame(latent_np, columns=[f"latent_{i}" for i in range(d)])
            csv_path = f"artifacts/sweep/B3_pca_{split}_d{d}_seed{seed}.csv"
            df.to_csv(csv_path, index=False)

    print(f"  [B3] d={d:2d} | seed={seed} → Best Val Loss: {best_val_loss:.6f} | Latent features salvate")
    return {"model": "B3", "d": d, "seed": seed, "best_val_loss": best_val_loss}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    os.makedirs("artifacts/sweep", exist_ok=True)

    results_log = []

    for d in DIMS:
        for seed in SEEDS:

            print(f"\n[B2] Avvio training → d={d}, seed={seed}")
            results_log.append(train_vanilla_ae(d, seed))

            print(f"\n[B3] Avvio training → d={d}, seed={seed}")
            results_log.append(train_regularized_ae(d, seed))

    df = pd.DataFrame(results_log)
    df = df.sort_values(["model", "d"], ascending=[True, False]).reset_index(drop=True)
    df.to_csv("artifacts/ae_training_report.csv", index=False)

    print("\n" + "=" * 60)
    print("Training B2 e B3 completato. Artifact generati:")
    print(df.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()