"""
week8_evaluation.py

Valutazione PULITA (nessuna perturbazione dell'input) di tutti e tre i
compressori (B1, B2, B3) su tutti gli split (train, val, test), usando il
VQC specifico per compressore addestrato da train_vqc_production.py.

Aggiornato per l'architettura con DATA RE-UPLOADING: il checkpoint non
contiene più feature_selector (rimosso — il circuito consuma direttamente
il vettore d-dim scalato/paddato). Il checkpoint include ora anche
min_vec/max_vec (lo scaler fittato sul train al momento dell'addestramento)
per applicare identicamente la stessa trasformazione in fase di valutazione.

Output: artifacts/week8_evaluation_report.csv
    colonne: dim, seed, compression, split, accuracy, macro_f1
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from qiskit_aer.primitives import EstimatorV2
from sklearn.metrics import accuracy_score, f1_score

from hybrid_engine import DirectVQC
from quantum_model import QuantumPipeline
from train_vqc_production import (
    N_QUBITS, N_CLASSES, N_LAYERS, COMPRESSOR_PATHS,
    apply_scaler, pad_features,
)


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
SPLITS      = ['train', 'val', 'test']
DIMENSIONS  = [32, 16, 8, 4]
SEEDS       = [11, 17, 29]
COMPRESSORS = ['B1', 'B2', 'B3']
MAX_WORKERS = 4
BATCH_SIZE  = 256


# ---------------------------------------------------------------------------
# Caricamento pipeline VQC specifica per compressore
#
# Checkpoint aggiornato: solo 'vqc' + 'classifier' (no feature_selector),
# più 'min_vec'/'max_vec' — lo scaler fittato sul train durante
# l'addestramento, necessario per applicare la stessa trasformazione qui.
# ---------------------------------------------------------------------------
def load_vqc_pipeline(d: int, seed: int, compressor: str, device):
    """
    Ricostruisce e carica vqc + classifier dal checkpoint prodotto da
    train_vqc_production.py, insieme allo scaler (min_vec, max_vec)
    fittato sul train al momento dell'addestramento.

    Returns:
        (vqc, classifier, min_vec, max_vec, n_encoding_padded)

    Raises:
        FileNotFoundError se il checkpoint non esiste.
    """
    ckpt_path  = f"experiments/models/best_vqc_{compressor}_d{d}_seed{seed}.pth"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    q_pipeline = QuantumPipeline(n_qubits=N_QUBITS, d_latent=d, n_layers=N_LAYERS)
    circuit    = q_pipeline.build_circuit()
    vqc = DirectVQC(
        circuit     = circuit,
        features_pv = q_pipeline.features,
        weights_pv  = q_pipeline.weights,
        n_qubits    = N_QUBITS,
        estimator   = EstimatorV2(),
    ).to(device)
    vqc.load_state_dict(checkpoint['vqc'])

    classifier = nn.Linear(N_QUBITS, N_CLASSES).to(device)
    classifier.load_state_dict(checkpoint['classifier'])

    vqc.eval()
    classifier.eval()

    return vqc, classifier, checkpoint['min_vec'], checkpoint['max_vec'], q_pipeline.n_encoding_padded


def vqc_pipeline_predict(vqc, classifier, u_scaled: torch.Tensor) -> torch.Tensor:
    """Pipeline di inferenza: feature già scalate/paddate → VQC → classifier → logits."""
    with torch.no_grad():
        q_out  = vqc(u_scaled)
        logits = classifier(q_out)
    return logits


# ---------------------------------------------------------------------------
# Lettura feature pulite — stessa convenzione di path di train_vqc_production.py
# ---------------------------------------------------------------------------
def load_clean_features(compressor: str, split: str, d: int, seed: int) -> tuple:
    """Legge feature d-dim RAW (non scalate) dal CSV del compressore."""
    path = COMPRESSOR_PATHS[compressor].format(split=split, d=d, seed=seed)
    df   = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c != 'label']
    X = df[feat_cols].values.astype(np.float32)
    y = df['label'].values.astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Valutazione — un compressore, una tripla (d, seed), tutti gli split
# ---------------------------------------------------------------------------
def run_clean_eval(d: int, seed: int, compressor: str) -> list:
    torch.set_num_threads(1)
    device = torch.device("cpu")

    try:
        vqc, classifier, min_vec, max_vec, target_width = load_vqc_pipeline(
            d, seed, compressor, device
        )
    except FileNotFoundError as exc:
        print(f"  [SKIP {compressor} d={d} s={seed}] VQC mancante: {exc}", flush=True)
        return []

    results = []

    for split in SPLITS:
        try:
            X, y = load_clean_features(compressor, split, d, seed)
        except FileNotFoundError as exc:
            print(f"  [SKIP {compressor}/{split} d={d} s={seed}] CSV mancante: {exc}", flush=True)
            continue

        # Stessa trasformazione usata in training: scaler fittato sul train
        # (min_vec/max_vec dal checkpoint) + padding a n_encoding_padded.
        X_scaled = pad_features(apply_scaler(X, min_vec, max_vec), target_width)
        u = torch.tensor(X_scaled, dtype=torch.float32, device=device)

        all_preds = []
        for i in range(0, len(u), BATCH_SIZE):
            batch  = u[i:i + BATCH_SIZE]
            logits = vqc_pipeline_predict(vqc, classifier, batch)
            preds  = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

        acc = accuracy_score(y, all_preds)
        f1  = f1_score(y, all_preds, average='macro', zero_division=0)

        print(
            f"  [{compressor}] d={d:2d} | seed={seed} | split={split} | "
            f"Acc: {acc:.4f} | F1: {f1:.4f}",
            flush=True,
        )

        results.append({
            'dim': d, 'seed': seed, 'compression': compressor,
            'split': split, 'accuracy': acc, 'macro_f1': f1,
        })

    return results


def run_job(job: tuple) -> list:
    d, seed, compressor = job
    print(f"Valutazione pulita: d={d}, seed={seed}, comp={compressor}", flush=True)
    return run_clean_eval(d, seed, compressor)


def main():
    import multiprocessing
    multiprocessing.freeze_support()

    os.makedirs("artifacts", exist_ok=True)

    jobs = [(d, s, c) for d in DIMENSIONS for s in SEEDS for c in COMPRESSORS]
    max_workers = min(len(jobs), MAX_WORKERS)
    print(f"[INFO] Avvio {len(jobs)} job su {max_workers} processi paralleli (nessuna perturbazione)...\n")

    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            d, s, comp = futures[future]
            try:
                rows = future.result()
                all_results.extend(rows)
                print(f"[OK] d={d} | seed={s} | comp={comp} → {len(rows)} risultati", flush=True)
            except Exception as e:
                print(f"[ERROR] d={d} | seed={s} | comp={comp} → {e}", flush=True)

    if not all_results:
        print("[WARN] Nessun risultato disponibile. Controllare gli artifact.")
        return

    df = pd.DataFrame(all_results)
    df = df.sort_values(
        ["compression", "dim", "seed", "split"], ascending=[True, False, True, True]
    ).reset_index(drop=True)
    df.to_csv("artifacts/week8_evaluation_report.csv", index=False)

    print("\nValutazione pulita completata. Artifact generati.")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()