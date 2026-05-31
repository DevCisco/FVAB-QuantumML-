import joblib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from medmnist import OCTMNIST
from pca_res_compressors import ResNetCompressor
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

from unsupervised_models import AS_RGB, VanillaAE, RegularizedAE
from vqc_fewshot_engine import FewShotVQC


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
SPLITS     = ['train', 'val', 'test']
BACKBONE   = 'resnet'


# ---------------------------------------------------------------------------
# Trasformazione rumore gaussiano
# ---------------------------------------------------------------------------
class AddGaussianNoise:
    """
    Aggiunge rumore gaussiano a un tensore immagine.
    Applicata PRIMA di Normalize: opera su valori raw [0,1] e la
    normalizzazione ImageNet viene applicata su dati coerenti.
    """

    def __init__(self, mean: float = 0., std: float = 0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)

    def __repr__(self) -> str:
        return f"AddGaussianNoise(mean={self.mean}, std={self.std})"


# ---------------------------------------------------------------------------
# B1: inferenza su feature pre-calcolate (train / val / test)
# ---------------------------------------------------------------------------
def run_b1_test(latent_dim: int, seed: int, split: str) -> dict | None:
    """
    Esegue il test di robustezza per B1 (PCA sklearn) su tutti e tre gli split.

    B1 carica le feature già compresse da CSV (prodotte da test.py fase 4).
    Path: artifacts/sweep/B1_pca_{split}_d{latent_dim}_seed{seed}.csv

    Il VQC viene caricato da:
        artifacts/checkpoints/vqc_d{latent_dim}_seed{seed}_best.pth

    Args:
        latent_dim (int): dimensione PCA / bottleneck (4/8/16/32).
        seed       (int): seed del VQC artifact.
        split      (str): split su cui eseguire il test (train/val/test).

    Returns:
        dict con chiavi: dim, seed, compression, split, accuracy, macro_f1.
        None se un artifact è mancante.
    """
    torch.set_num_threads(1)  # worker parallelo — evita contese BLAS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # — Caricamento VQC -------------------------------------------------------
    try:
        vqc = FewShotVQC(d_latent=latent_dim)
        vqc.load_state_dict(
            torch.load(
                f"artifacts/checkpoints/vqc_d{latent_dim}_seed{seed}_best.pth",
                map_location=device,
            ),
            strict=False,
        )
        vqc = vqc.to(device).eval()
    except FileNotFoundError as exc:
        print(f"  [SKIP B1] Artifact VQC mancante: {exc}", flush=True)
        return None

    results = []

    for split in SPLITS:
        # — Caricamento feature pre-compresse da CSV ---------------------------
        csv_path = f"artifacts/sweep/B1_pca_{split}_d{latent_dim}_seed{seed}.csv"
        try:
            df        = pd.read_csv(csv_path)
            features  = df.iloc[:, :-1].values.astype(np.float32)  # tutte le colonne tranne ultima
            labels    = df.iloc[:, -1].values.astype(int) if 'label' in df.columns else None
        except FileNotFoundError as exc:
            print(f"  [SKIP B1/{split}] CSV mancante: {exc}", flush=True)
            continue

        if labels is None:
            print(f"  [SKIP B1/{split}] Colonna 'label' non trovata in CSV", flush=True)
            continue

        # — Inferenza VQC direttamente sulle feature compresse ------------------
        compressed = torch.tensor(features, dtype=torch.float32).to(device)

        with torch.no_grad():
            # Inferenza a batch per evitare OOM su split grandi
            all_preds = []
            batch_size = 256
            for i in range(0, len(compressed), batch_size):
                batch  = compressed[i: i + batch_size]
                logits = vqc(batch)
                preds  = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)

        acc = accuracy_score(labels, all_preds)
        f1  = f1_score(labels, all_preds, average='macro')

        print(
            f"  [B1] d={latent_dim:2d} | seed={seed} | split={split} | "
            f"Acc: {acc:.4f} | F1: {f1:.4f}",
            flush=True,
        )

        results.append({
            'dim':         latent_dim,
            'seed':        seed,
            'compression': 'B1',
            'split':       split,
            'accuracy':    acc,
            'macro_f1':    f1,
        })

    return results if results else None


# ---------------------------------------------------------------------------
# B2 / B3: inferenza con backbone live su split 'test' con rumore gaussiano
# ---------------------------------------------------------------------------
def run_ae_test(latent_dim: int, seed: int, compression_type: str) -> dict | None:
    """
    Esegue il test di robustezza per B2 (VanillaAE) o B3 (RegularizedAE).

    Usa il backbone ResNet18 live su immagini OCTMNIST con rumore gaussiano
    (split 'test' — coerente con il protocollo di audit Week 8).

    Args:
        latent_dim       (int): dimensione bottleneck AE (4/8/16/32).
        seed             (int): seed dell'artifact AE.
        compression_type (str): 'B2' | 'B3'.

    Returns:
        dict con chiavi: dim, seed, compression, split, accuracy, macro_f1.
        None se un artifact è mancante.
    """
    torch.set_num_threads(1)  # worker parallelo — evita contese BLAS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # — Backbone ResNet18 congelato -------------------------------------------
    backbone = ResNetCompressor(data_flag='octmnist', as_rgb=AS_RGB)
    backbone = backbone.to(device).eval()

    # — Pipeline trasformazione -----------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.1),       # rumore PRIMA di Normalize
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = OCTMNIST(split='test', transform=transform, download=True, as_rgb=AS_RGB)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # — Caricamento compressore AE --------------------------------------------
    try:
        if compression_type == 'B2':
            compressor = VanillaAE(d_latent=latent_dim).to(device)
            compressor.load_state_dict(
                torch.load(
                    f"artifacts/sweep/B2_pca_test_d{latent_dim}_seed{seed}.csv",
                    map_location=device,
                )
            )
        else:  # B3
            compressor = RegularizedAE(d_latent=latent_dim).to(device)
            compressor.load_state_dict(
                torch.load(
                    f"artifacts/sweep/B3_pca_test_d{latent_dim}_seed{seed}.csv",
                    map_location=device,
                )
            )
        compressor.eval()

    except FileNotFoundError as exc:
        print(f"  [SKIP {compression_type}] Artifact AE mancante: {exc}", flush=True)
        return None

    # — Caricamento VQC -------------------------------------------------------
    try:
        vqc = FewShotVQC(d_latent=latent_dim)
        vqc.load_state_dict(
            torch.load(
                f"artifacts/checkpoints/vqc_d{latent_dim}_seed{seed}_best.pth",
                map_location=device,
            ),
            strict=False,
        )
        vqc = vqc.to(device).eval()
    except FileNotFoundError as exc:
        print(f"  [SKIP {compression_type}] Artifact VQC mancante: {exc}", flush=True)
        return None

    # — Inferenza -------------------------------------------------------------
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images   = images.to(device)
            features = backbone(images)            # → (B, 512)
            _, compressed = compressor(features)   # → (B, latent_dim)
            logits = vqc(compressed)
            preds  = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().squeeze().numpy())

    acc = accuracy_score(all_targets, all_preds)
    f1  = f1_score(all_targets, all_preds, average='macro')

    print(
        f"  [{compression_type}] d={latent_dim:2d} | seed={seed} | split=test | "
        f"Acc: {acc:.4f} | F1: {f1:.4f}",
        flush=True,
    )

    return {
        'dim':         latent_dim,
        'seed':        seed,
        'compression': compression_type,
        'split':       'test',
        'accuracy':    acc,
        'macro_f1':    f1,
    }


# ---------------------------------------------------------------------------
# Worker — chiamato da ProcessPoolExecutor
# ---------------------------------------------------------------------------
def run_job(job: tuple) -> list:
    """
    Dispatcher per worker parallelo.

    Args:
        job (tuple): (latent_dim, seed, compression_type)

    Returns:
        Lista di dict risultato (B1 restituisce 3 dict, uno per split;
        B2/B3 restituiscono 1 dict per split 'test').
    """
    latent_dim, seed, compression_type = job

    print(
        f"Audit Week 8: d={latent_dim}, seed={seed}, comp={compression_type}",
        flush=True,
    )

    if compression_type == 'B1':
        result = run_b1_test(latent_dim, seed)
        # run_b1_test restituisce una lista di dict (uno per split) o None
        return result if result is not None else []
    else:
        result = run_ae_test(latent_dim, seed, compression_type)
        return [result] if result is not None else []


# ---------------------------------------------------------------------------
# Entry point — parallelizzazione Windows-safe
# ---------------------------------------------------------------------------
def main():
    os.makedirs("artifacts", exist_ok=True)

    dimensions       = [32, 16, 8, 4]
    robustness_seeds = [11, 17, 29]

    # 36 job totali: 4 dim × 3 seed × 3 comp
    jobs = [
        (d, s, comp)
        for d    in dimensions
        for s    in robustness_seeds
        for comp in ['B1', 'B2', 'B3']
    ]

    # Su Windows non superare i core fisici disponibili.
    # Ogni job è CPU-bound (backbone + inferenza PyTorch CPU).
    max_workers = min(len(jobs), os.cpu_count() or 1)
    print(f"[INFO] Avvio {len(jobs)} job su {max_workers} processi paralleli...\n")

    all_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_job, job): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            d, s, comp = job
            try:
                rows = future.result()   # lista di dict
                all_results.extend(rows)
                print(f"[OK] d={d} | seed={s} | comp={comp} → {len(rows)} risultati", flush=True)
            except Exception as e:
                print(f"[ERROR] d={d} | seed={s} | comp={comp} → {e}", flush=True)

    if not all_results:
        print("[WARN] Nessun risultato disponibile. Controllare gli artifact.")
        return

    df = pd.DataFrame(all_results)
    df = df.sort_values(
        ["compression", "dim", "seed", "split"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)

    df.to_csv("artifacts/week8_robustness_report.csv", index=False)
    print("\nFreeze Week 8 completato. Artifact generati.")
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Obbligatorio su Windows: senza questo guard ogni worker
# rilancerebbe main() ricorsivamente (freeze_support error).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()