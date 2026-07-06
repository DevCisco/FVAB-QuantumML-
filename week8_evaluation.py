"""
week8_evaluation.py

Valutazione PULITA (nessuna perturbazione dell'input) di tutti e tre i
compressori (B1, B2, B3) su tutti gli split (train, val, test), usando il
VQC specifico per compressore addestrato da train_vqc_production.py.

Sostituisce week8_robustness.py: quello script iniettava rumore gaussiano
sulle immagini (AddGaussianNoise) per un audit di robustezza — la
perturbazione dell'input non è richiesta per questa consegna. Questo
script mantiene la stessa struttura di valutazione (feature_selector →
VQC → classifier) ma opera solo su feature pulite, pre-calcolate dai CSV
prodotti da test.py (B1) e b2_b3_training.py (B2, B3). Nessun backbone
live, nessuna trasformazione immagine, nessuna dipendenza da
torchvision.transforms.

Colma inoltre un gap della versione precedente: prima solo B1 aveva una
valutazione pulita su tutti e tre gli split (via run_b1_test); B2/B3
avevano solo val+test, recuperati da production_summary.csv. Qui tutti
e tre i compressori sono valutati simmetricamente su train/val/test —
108 righe totali (3 compressori × 4 dim × 3 seed × 3 split).

Output: artifacts/week8_evaluation_report.csv
    colonne: dim, seed, compression, split, accuracy, macro_f1

Nota: per B1, questi numeri devono coincidere esattamente con quelli già
noti da production_summary.csv (val/test) — è lo stesso identico VQC
valutato sulle stesse identiche feature, solo con un percorso di codice
indipendente. Una discrepanza indicherebbe un bug in uno dei due script.
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
    scale_features, N_QUBITS, N_CLASSES, N_LAYERS, COMPRESSOR_PATHS,
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
# Identica alla versione in week8_robustness.py: il checkpoint prodotto da
# train_vqc_production.py è un dict con tre state_dict separati
# (feature_selector, vqc, classifier), non un unico modello flat.
# ---------------------------------------------------------------------------
def load_vqc_pipeline(d: int, seed: int, compressor: str, device):
    """
    Ricostruisce e carica feature_selector + vqc + classifier dal checkpoint
    prodotto da train_vqc_production.py per una tripla (d, seed, compressor).

    Raises:
        FileNotFoundError se il checkpoint non esiste.
    """
    ckpt_path = f"experiments/models/best_vqc_{compressor}_d{d}_seed{seed}.pth"
    checkpoint = torch.load(ckpt_path, map_location=device)

    feature_selector = nn.Linear(d, N_QUBITS, bias=False).to(device)
    feature_selector.load_state_dict(checkpoint['feature_selector'])

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

    feature_selector.eval()
    vqc.eval()
    classifier.eval()

    return feature_selector, vqc, classifier


def vqc_pipeline_predict(feature_selector, vqc, classifier,
                          u_d: torch.Tensor) -> torch.Tensor:
    """
    Pipeline di inferenza completa: feature d-dim → proiezione N_QUBITS-dim
    → scaling [0,π] → VQC → classifier → logits. Nessuna perturbazione
    applicata in nessun punto della catena.
    """
    with torch.no_grad():
        u_4      = feature_selector(u_d)
        u_scaled = scale_features(u_4)
        q_out    = vqc(u_scaled)
        logits   = classifier(q_out)
    return logits


# ---------------------------------------------------------------------------
# Lettura feature pulite — stessa convenzione di path di train_vqc_production.py
# ---------------------------------------------------------------------------
def load_clean_features(compressor: str, split: str, d: int, seed: int) -> tuple:
    """
    Legge feature d-dim pulite (nessuna perturbazione) dal CSV del compressore.
    Riusa COMPRESSOR_PATHS di train_vqc_production.py — nessuna duplicazione
    della logica di path tra i due script.
    """
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
    """
    Valuta il VQC specifico per compressore su train/val/test, senza
    alcuna perturbazione dell'input.

    Returns:
        Lista di dict (uno per split) con chiavi:
        dim, seed, compression, split, accuracy, macro_f1.
        Lista vuota se un artifact è mancante.
    """
    torch.set_num_threads(1)
    device = torch.device("cpu")

    try:
        feature_selector, vqc, classifier = load_vqc_pipeline(d, seed, compressor, device)
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

        u = torch.tensor(X, dtype=torch.float32, device=device)

        all_preds = []
        for i in range(0, len(u), BATCH_SIZE):
            batch  = u[i:i + BATCH_SIZE]
            logits = vqc_pipeline_predict(feature_selector, vqc, classifier, batch)
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
            'dim':         d,
            'seed':        seed,
            'compression': compressor,
            'split':       split,
            'accuracy':    acc,
            'macro_f1':    f1,
        })

    return results


# ---------------------------------------------------------------------------
# Worker — chiamato da ProcessPoolExecutor
# ---------------------------------------------------------------------------
def run_job(job: tuple) -> list:
    d, seed, compressor = job
    print(f"Valutazione pulita: d={d}, seed={seed}, comp={compressor}", flush=True)
    return run_clean_eval(d, seed, compressor)


# ---------------------------------------------------------------------------
# Entry point — parallelizzazione Windows-safe
# ---------------------------------------------------------------------------
def main():
    import multiprocessing
    multiprocessing.freeze_support()

    os.makedirs("artifacts", exist_ok=True)

    # 36 job: 4 dim × 3 seed × 3 compressori. Ogni job valuta 3 split
    # internamente → 108 righe totali nel report finale.
    jobs = [
        (d, s, c)
        for d in DIMENSIONS
        for s in SEEDS
        for c in COMPRESSORS
    ]

    max_workers = min(len(jobs), MAX_WORKERS)
    print(f"[INFO] Avvio {len(jobs)} job su {max_workers} processi paralleli "
          f"(nessuna perturbazione)...\n")

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
        ["compression", "dim", "seed", "split"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)

    df.to_csv("artifacts/week8_evaluation_report.csv", index=False)
    print("\nValutazione pulita completata. Artifact generati.")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()