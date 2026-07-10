"""
execute_week_7.py

Sweep few-shot: LogisticRegression + VQC su frazioni decrescenti del
training set (25%, 10%, 5%), per tutti e tre i compressori B1/B2/B3,
dimensioni D=[8,4], seed=[11].

FIX rispetto alla versione precedente: quella dipendeva da
vqc_fewshot_engine.train_vqc, un modulo esterno mai allineato ai fix
applicati a train_vqc_production.py (rimozione di feature_selector,
introduzione del data re-uploading, scaler fit-on-train) — avrebbe
addestrato un VQC strutturalmente DIVERSO da quello usato ovunque altrove
nel progetto, rendendo il confronto few-shot non comparabile col resto
dei risultati. Inoltre copriva solo B1 (non B2/B3) e misurava accuracy su
soli 400 campioni di test invece di macro-F1 sul test set fisso completo.

Questa versione riusa DIRETTAMENTE i building block già validati di
train_vqc_production.py (QuantumPipeline, DirectVQC, scaler fit-on-train,
NFT, budget scalato) — stessa architettura, stesso encoding, stesso
ansatz, stesso ottimizzatore del benchmark principale. L'unica differenza
è che il pool di training per il campionamento dei batch NFT viene
ristretto al sottoinsieme few-shot stratificato, invece che all'intero
training set canonico.

Protocollo (few-shot cambia SOLO il training set — validation e test
restano i set fissi standard, invariati rispetto al benchmark principale).

Output: artifacts/fewshot_final_results.csv
    colonne: compressor, d, seed, fraction, macro_f1_lr, macro_f1_vqc
"""

import multiprocessing
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor, as_completed
from qiskit_aer.primitives import EstimatorV2
from qiskit_algorithms.optimizers import NFT
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score as sk_f1

from fewshot_sampler import get_stratified_fraction_indices, save_fewshot_manifest
from hybrid_engine import DirectVQC
from quantum_model import QuantumPipeline
from train_vqc_production import (
    N_QUBITS, N_CLASSES, N_LAYERS, COMPRESSOR_PATHS,
    fit_scaler, apply_scaler, pad_features, get_max_evals_nft,
    compute_class_weights, sample_balanced_batch,
    get_trainable_params, set_trainable_params, make_loss_fn,
    evaluate_on_features,
)


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
SEEDS       = [11]   # seed=23 nel documento è un refuso confermato dal docente
FRACTIONS   = [0.25, 0.10, 0.05]
DIMS        = [8, 4]
COMPRESSORS = ['B1', 'B2', 'B3']

EPOCHS            = 10
PATIENCE          = 3
SAMPLES_PER_CLASS = 8
MAX_WORKERS       = 4

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Lettura CSV — stessa convenzione di path di train_vqc_production.py
# ---------------------------------------------------------------------------
def load_raw(compressor: str, split: str, d: int, seed: int) -> tuple:
    path = COMPRESSOR_PATHS[compressor].format(split=split, d=d, seed=seed)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File non trovato: {path}\n"
            f"Per B1: eseguire test.py\n"
            f"Per B2/B3: eseguire b2_b3_training.py"
        )
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c != 'label']
    X = df[feat_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Worker — una tripla (compressor, d, seed), tutte le frazioni
# ---------------------------------------------------------------------------
def run_single(compressor: str, d: int, seed: int, fractions: list) -> list:
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"]      = "1"
    os.environ["MKL_NUM_THREADS"]      = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    results = []

    X_train_full, y_train_full = load_raw(compressor, 'train', d, seed)
    X_val,  y_val  = load_raw(compressor, 'val',  d, seed)
    X_test, y_test = load_raw(compressor, 'test', d, seed)

    for f in fractions:
        print(f"\n>>> RUN: {compressor} d={d} seed={seed} fraction={int(f*100)}%", flush=True)

        # — Subset few-shot stratificato (SOLO sul training) --------------------
        idx = get_stratified_fraction_indices(y_train_full, f, seed)
        save_fewshot_manifest(idx, f, seed, d)

        X_fs = X_train_full[idx]
        y_fs = y_train_full[idx]

        # — LogisticRegression sullo stesso subset few-shot -----------------------
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_fs, y_fs)
        preds_lr    = clf.predict(X_test)
        macro_f1_lr = float(sk_f1(y_test, preds_lr, average='macro', zero_division=0))

        # — VQC: stessa architettura di train_vqc_production.py -------------------
        q_pipeline = QuantumPipeline(n_qubits=N_QUBITS, d_latent=d, n_layers=N_LAYERS)
        circuit    = q_pipeline.build_circuit()
        vqc = DirectVQC(
            circuit     = circuit,
            features_pv = q_pipeline.features,
            weights_pv  = q_pipeline.weights,
            n_qubits    = N_QUBITS,
            estimator   = EstimatorV2(),
        ).to(DEVICE)
        classifier = nn.Linear(N_QUBITS, N_CLASSES).to(DEVICE)
        modules    = [vqc, classifier]

        # Scaler fittato SOLO sul subset few-shot — coerente col principio
        # "fit solo su train" applicato anche quando il train è ristretto.
        min_vec, max_vec = fit_scaler(X_fs)
        target_width      = q_pipeline.n_encoding_padded

        X_fs_s   = pad_features(apply_scaler(X_fs,   min_vec, max_vec), target_width)
        X_val_s  = pad_features(apply_scaler(X_val,  min_vec, max_vec), target_width)
        X_test_s = pad_features(apply_scaler(X_test, min_vec, max_vec), target_width)

        u_pool = torch.tensor(X_fs_s,   dtype=torch.float32, device=DEVICE)
        u_val  = torch.tensor(X_val_s,  dtype=torch.float32, device=DEVICE)
        u_test = torch.tensor(X_test_s, dtype=torch.float32, device=DEVICE)

        class_weights = compute_class_weights(y_fs).to(DEVICE)
        criterion     = nn.CrossEntropyLoss(weight=class_weights)

        n_total   = sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad)
        max_evals = get_max_evals_nft(n_total)
        optimizer = NFT(maxfev=max_evals)
        epoch_rng = np.random.default_rng(seed)

        best_val_f1  = 0.0
        best_test_f1 = 0.0
        patience_ctr = 0

        # Stesso loop epoch/early-stopping del benchmark principale. Il pool
        # per il campionamento batch è ristretto al subset few-shot: se una
        # classe ha meno di SAMPLES_PER_CLASS esempi disponibili,
        # sample_balanced_batch ricampiona con replace=True (stesso
        # comportamento generico già usato in train_vqc_production.py).
        for epoch in range(EPOCHS):
            u_batch, y_batch_np = sample_balanced_batch(
                u_pool, y_fs, SAMPLES_PER_CLASS, epoch_rng
            )
            y_batch_t = torch.tensor(y_batch_np, dtype=torch.long, device=DEVICE)

            loss_fn = make_loss_fn(modules, u_batch, y_batch_t, criterion)
            result  = optimizer.minimize(fun=loss_fn, x0=get_trainable_params(modules))
            set_trainable_params(modules, result.x)

            _, _, val_f1,  _ = evaluate_on_features(modules, u_val,  y_val,  criterion, DEVICE)
            _, _, test_f1, _ = evaluate_on_features(modules, u_test, y_test, criterion, DEVICE)

            if val_f1 > best_val_f1:
                best_val_f1  = val_f1
                best_test_f1 = test_f1
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    break

        print(
            f"    {compressor} d={d} f={int(f*100)}% | "
            f"macro-F1 LR: {macro_f1_lr:.4f} | macro-F1 VQC: {best_test_f1:.4f}",
            flush=True,
        )
        results.append({
            "compressor":   compressor,
            "d":            d,
            "seed":         seed,
            "fraction":     f,
            "macro_f1_lr":  round(macro_f1_lr, 6),
            "macro_f1_vqc": round(best_test_f1, 6),
        })

    return results


# ---------------------------------------------------------------------------
# Worker wrapper — chiamato da ProcessPoolExecutor
# ---------------------------------------------------------------------------
def run_job(job: tuple) -> list:
    compressor, d, seed = job
    try:
        return run_single(compressor, d, seed, FRACTIONS)
    except FileNotFoundError as e:
        print(f"[SKIP] {compressor} d={d} seed={seed} → {e}", flush=True)
        return []


# ---------------------------------------------------------------------------
# Entry point — parallelizzazione Windows-safe
# ---------------------------------------------------------------------------
def main():
    multiprocessing.freeze_support()
    os.makedirs("artifacts", exist_ok=True)

    jobs = [(c, d, s) for c in COMPRESSORS for d in DIMS for s in SEEDS]

    max_workers = min(len(jobs), MAX_WORKERS)
    print(f"[INFO] Avvio {len(jobs)} run su {max_workers} processi paralleli...")
    print(f"[INFO] Compressori: {COMPRESSORS} | D: {DIMS} | Seed: {SEEDS} | Frazioni: {FRACTIONS}\n")

    all_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_job, job): job
            for job in jobs
        }
        for future in as_completed(futures):
            c, d, s = futures[future]
            try:
                partial = future.result()
                all_results.extend(partial)
                print(f"[OK] {c} d={d} seed={s} completato ({len(partial)} run)", flush=True)
            except Exception as e:
                print(f"[ERROR] {c} d={d} seed={s} → {e}", flush=True)

    if not all_results:
        print("\n[WARNING] Nessun risultato disponibile. Verifica gli errori sopra.")
        return

    df = pd.DataFrame(all_results)
    df = df.sort_values(
        ["compressor", "d", "seed", "fraction"],
        ascending=[True, False, True, False],
    ).reset_index(drop=True)
    df.to_csv("artifacts/fewshot_final_results.csv", index=False)

    print("\n[DONE] Sweep few-shot completato.")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()