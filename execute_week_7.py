"""
execute_week_7.py
=================
Esegue il protocollo few-shot (LR + VQC) su tutte le coppie (d, seed, fraction).

Ottimizzazione per 4 core fisici Windows
-----------------------------------------
Layout processi/thread scelto per saturare i core senza oversubscription:

    max_workers = 2   (DIMS=[8,4] × SEEDS=[11] → 2 job indipendenti)

    Per ogni worker:
        torch threads    = 1   (torch.set_num_threads)
        OMP threads      = 1   (os.environ)
        MKL threads      = 1   (os.environ)
        Aer threads      = 2   (configurato in vqc_fewshot_engine._make_estimator)

    Totale thread attivi simultaneamente: 2 worker × 2 Aer thread = 4 = core fisici

I due job (d=8 e d=4) girano in parallelo, ciascuno su 2 core Aer.
Ogni job processa le fractions [0.25, 0.10, 0.05] in sequenza (le fractions
non sono parallelizzabili indipendentemente perché condividono i dati caricati).
"""

import os
import numpy as np
import torch
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fewshot_sampler import get_stratified_fraction_indices, save_fewshot_manifest
from vqc_fewshot_engine import train_vqc

# Numero di worker fisso a 2: con DIMS=[8,4] e SEEDS=[11] ci sono esattamente
# 2 job indipendenti. Aumentare oltre 2 non porta benefici e riduce i thread
# disponibili per il simulatore Aer in ogni worker.
MAX_WORKERS = 2


def _set_single_thread_env():
    """
    Imposta tutti i layer di parallelismo a 1 thread per il processo corrente.
    Va chiamato all'inizio di ogni worker (spawn non eredita le env vars del padre
    impostate dopo l'avvio del processo su Windows).

    torch.set_num_threads agisce su PyTorch.
    Le variabili d'ambiente agiscono su OpenMP (numpy, scipy, sklearn) e MKL.
    Devono essere impostate PRIMA di importare i moduli che le leggono, ma
    poiché siamo già in un worker spawn pulito, l'impostazione è efficace.
    """
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"]     = "1"
    os.environ["MKL_NUM_THREADS"]     = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def run_single(d: int, s: int, fractions: list) -> list:
    """
    Worker top-level: esegue tutte le fractions per la coppia (d, seed).

    Funzione top-level (non lambda, non annidata): obbligatorio per la
    serializzazione pickle con il metodo spawn di Windows.

    Returns:
        list[dict]: risultati per ogni (d, seed, fraction).
    """
    _set_single_thread_env()

    results = []

    # Lettura CSV prodotti da test.py con la convenzione:
    #   {split}_ids_dimensione{d}_seed{seed}_.csv
    # Percorso corretto allineato a save_pca_features_csv in test.py.
    train_path = (
        f"artifacts/sweep/b1_pca_train_d{d}_seed{s}.csv"
    )
    test_path = (
        f"artifacts/sweep/b1_pca_test_d{d}_seed{s}.csv"
    )

    try:
        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)
    except FileNotFoundError as exc:
        print(f"[ERROR] CSV non trovato: {exc}", flush=True)
        return results

    feat_cols = [c for c in train_df.columns if c != 'label']

    X_full  = train_df[feat_cols].values.astype(np.float32)
    y_full  = train_df['label'].values.ravel()
    # Test set completo — nessun troncamento per non distorcere la metrica
    X_test  = torch.tensor(test_df[feat_cols].values.astype(np.float32))
    y_test  = test_df['label'].values.ravel()

    for f in fractions:
        print(f"\n>>> RUN: d={d} | Seed={s} | Fraction={int(f*100)}%", flush=True)

        # 1. Subset stratificato
        idx = get_stratified_fraction_indices(y_full, f, s)
        save_fewshot_manifest(idx, f, s, d)

        X_train_fs = torch.tensor(X_full[idx]).float()
        y_train_fs = torch.tensor(y_full[idx]).long()

        # 2. Logistic Regression (baseline no-quantum)
        clf = LogisticRegression(C=0.5, max_iter=1000, random_state=s)
        clf.fit(X_train_fs.numpy(), y_train_fs.numpy())
        acc_lr = accuracy_score(y_test, clf.predict(X_test.numpy())) * 100

        # 3. VQC con EstimatorV2 nativo
        model = train_vqc(X_train_fs, y_train_fs, d)
        with torch.no_grad():
            preds   = model(X_test).argmax(1)
            acc_vqc = accuracy_score(y_test, preds.numpy()) * 100

        print(f"    ACC → LR: {acc_lr:.2f}% | VQC: {acc_vqc:.2f}%", flush=True)
        results.append({
            "d":        d,
            "seed":     s,
            "fraction": f,
            "acc_lr":   round(acc_lr,  4),
            "acc_vqc":  round(acc_vqc, 4),
        })

    return results


def main():
    SEEDS     = [11]               # estendibile a [11, 23]
    FRACTIONS = [0.25, 0.10, 0.05]
    DIMS      = [8, 4]

    jobs = [(d, s) for d in DIMS for s in SEEDS]

    # MAX_WORKERS=2: i 2 job usano ciascuno 2 thread Aer → 4 thread totali = 4 core
    print(
        f"[INFO] Avvio {len(jobs)} job su {MAX_WORKERS} processi paralleli "
        f"(2 thread Aer per worker, 4 core totali)...\n",
        flush=True,
    )

    all_results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_single, d, s, FRACTIONS): (d, s)
            for d, s in jobs
        }
        for future in as_completed(futures):
            d, s = futures[future]
            try:
                partial = future.result()
                all_results.extend(partial)
                print(
                    f"[OK] d={d}, seed={s} → {len(partial)} run completati",
                    flush=True,
                )
            except Exception as exc:
                print(f"[ERROR] d={d}, seed={s} → {exc}", flush=True)

    os.makedirs("artifacts", exist_ok=True)

    if not all_results:
        print("\n[WARNING] Nessun risultato. Verifica gli errori sopra.")
        return

    df = pd.DataFrame(all_results)
    df = df.sort_values(
        ["d", "seed", "fraction"],
        ascending=[False, True, False]
    ).reset_index(drop=True)
    df.to_csv("artifacts/fewshot_final_results.csv", index=False)
    print("\n[DONE] Policy few-shot completata.")
    print(df.to_string(index=False))


# Guard obbligatoria su Windows (spawn): senza di essa ogni worker
# rilancia main() ricorsivamente → loop infinito di processi.
if __name__ == "__main__":
    freeze_support()
    main()