"""
master_sweep_team_b.py

Confronto sistematico dei tre compressori B1/B2/B3 con LogisticRegression
su tutte le combinazioni (d, seed).

Prerequisiti — questi file devono esistere prima di eseguire questo script:
  • B1: artifacts/sweep/B1_pca_{split}_d{d}_seed{seed}.csv
          → generati da test.py
  • B2: artifacts/sweep/B2_pca_{split}_d{d}_seed{seed}.csv
          → generati da b2_b3_training.py
  • B3: artifacts/sweep/B3_pca_{split}_d{d}_seed{seed}.csv
          → generati da b2_b3_training.py

Metriche riportate per ogni combinazione (compressore, d, seed):
  • macro-F1 su test set (metrica principale, coerente con train_vqc_production.py)
  • accuracy su test set

Output:
  • artifacts/sweep/team_b_comparison.csv  — dettaglio per ogni (compressore, d, seed)
  • artifacts/sweep/team_b_summary.csv     — media e std per (compressore, d)

Il VQC non è incluso qui — è gestito da train_vqc_production.py.
Questo file confronta solo i compressori come estrattori di feature,
usando LogisticRegression come classificatore di riferimento.
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
DIMS         = [32, 16, 8, 4]
SEEDS        = [11, 17, 29]
COMPRESSORS  = ['B1', 'B2', 'B3']
ARTIFACTS    = "artifacts/sweep"

# Path CSV per ogni compressore — devono corrispondere esattamente a quelli
# prodotti da test.py (B1) e b2_b3_training.py (B2, B3).
CSV_PATHS = {
    'B1': "artifacts/sweep/B1_pca_{split}_d{d}_seed{seed}.csv",
    'B2': "artifacts/sweep/B2_pca_{split}_d{d}_seed{seed}.csv",
    'B3': "artifacts/sweep/B3_pca_{split}_d{d}_seed{seed}.csv",
}


# ---------------------------------------------------------------------------
# Lettura CSV
# ---------------------------------------------------------------------------
def load_features(compressor: str, split: str, d: int, seed: int) -> tuple:
    """
    Carica feature e label dal CSV del compressore specificato.

    Il formato è identico per B1, B2 e B3:
      colonne feat_*  → feature numeriche (d colonne)
      colonna label   → classe target (0-3)

    Args:
        compressor: 'B1' | 'B2' | 'B3'
        split:      'train' | 'val' | 'test'
        d:          dimensione latente (4/8/16/32)
        seed:       seed (11/17/29)

    Returns:
        X (ndarray float32, shape (N, d)), y (ndarray int, shape (N,))
    """
    path = CSV_PATHS[compressor].format(split=split, d=d, seed=seed)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File non trovato: {path}\n"
            f"Per B1: eseguire test.py\n"
            f"Per B2/B3: eseguire b2_b3_training.py"
        )

    df        = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c != 'label']
    X         = df[feat_cols].values.astype(np.float32)
    y         = df['label'].values.astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------
def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE per classificazione multiclasse con binning uniforme.

    Usa la probabilità massima predetta (top-1 confidence) come stima
    della fiducia del modello. Misura quanto le probabilità predette
    corrispondano alle frequenze empiriche di correttezza.

    Formula:
        ECE = Σ_b (|B_b| / n) · |acc(B_b) − conf(B_b)|

    dove B_b è l'insieme dei campioni il cui confidence cade nel bin b,
    acc(B_b) è l'accuratezza empirica nel bin e conf(B_b) è la fiducia media.

    Un modello perfettamente calibrato ha ECE = 0.
    Valori alti indicano overconfidence o underconfidence sistematica.

    Args:
        y_true  : label reali, shape (N,).
        y_proba : probabilità predette, shape (N, n_classes).
        n_bins  : numero di bin uniformi in [0, 1]. Default 10.

    Returns:
        ECE in [0, 1].
    """
    confidences = y_proba.max(axis=1)          # top-1 confidence per campione
    predictions = y_proba.argmax(axis=1)
    correct     = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Ultimo bin: include il boundary superiore (confidence == 1.0)
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)

        if mask.sum() == 0:
            continue

        bin_acc  = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_w    = mask.sum() / len(y_true)
        ece     += bin_w * abs(bin_acc - bin_conf)

    return float(ece)



# ---------------------------------------------------------------------------
# Valutazione con LogisticRegression
# ---------------------------------------------------------------------------
def evaluate_compressor(
    compressor: str,
    d:          int,
    seed:       int,
) -> dict:
    """
    Addestra LogisticRegression su train e valuta su test con quattro metriche.

    Metriche calcolate:
      • macro-F1          — media non pesata del F1 per classe
      • macro-AUROC       — area sotto la curva ROC, schema OvR, media su classi
      • balanced accuracy — media dei recall per classe (= macro-recall)
      • ECE               — Expected Calibration Error (top-1 confidence, 10 bin)
      • accuracy          — accuracy standard, a titolo di riferimento

    Usa il test set fisso (canonical val + canonical test MedMNIST),
    identico per tutti i seed — coerente con train_vqc_production.py.
    """
    X_train, y_train = load_features(compressor, 'train', d, seed)
    X_test,  y_test  = load_features(compressor, 'test',  d, seed)

    clf = LogisticRegression(
        max_iter    = 1000,
        random_state= seed,
        solver      = 'lbfgs'
    )
    clf.fit(X_train, y_train)

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)   # (N, 4) — necessario per AUROC e ECE

    macro_f1  = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
    macro_auc = float(roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'))
    bal_acc   = float(balanced_accuracy_score(y_test, y_pred))
    ece       = compute_ece(y_test, y_proba, n_bins=10)
    accuracy  = float(accuracy_score(y_test, y_pred))

    return {
        'compressor':   compressor,
        'd':            d,
        'seed':         seed,
        'macro_f1':     round(macro_f1,  6),
        'macro_auroc':  round(macro_auc, 6),
        'balanced_acc': round(bal_acc,   6),
        'ece':          round(ece,        6),
        'accuracy':     round(accuracy,   6),
    }



# ---------------------------------------------------------------------------
# Esperimento completo
# ---------------------------------------------------------------------------
def run_comparison() -> pd.DataFrame:
    """
    Esegue la valutazione per tutte le combinazioni (compressore, d, seed)
    e restituisce il DataFrame dei risultati dettagliati.
    """
    results = []
    n_total = len(COMPRESSORS) * len(DIMS) * len(SEEDS)
    done    = 0

    for compressor in COMPRESSORS:
        for d in DIMS:
            for seed in SEEDS:
                done += 1
                try:
                    row = evaluate_compressor(compressor, d, seed)
                    results.append(row)
                    print(
                        f"[{done:2d}/{n_total}] {compressor} d={d:2d} seed={seed} | "
                        f"F1: {row['macro_f1']:.4f} | "
                        f"AUROC: {row['macro_auroc']:.4f} | "
                        f"BalAcc: {row['balanced_acc']:.4f} | "
                        f"ECE: {row['ece']:.4f}",
                        flush=True,
                    )
                except FileNotFoundError as e:
                    print(f"[SKIP] {compressor} d={d} seed={seed} → {e}", flush=True)
                except Exception as e:
                    print(f"[ERROR] {compressor} d={d} seed={seed} → {e}", flush=True)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Riepilogo per (compressore, d)
# ---------------------------------------------------------------------------
def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola media e std di tutte le metriche su tutti i seed
    per ogni coppia (compressore, d).
    """
    summary = (
        df.groupby(['compressor', 'd'])
        .agg(
            macro_f1_mean    =('macro_f1',     'mean'),
            macro_f1_std     =('macro_f1',     'std'),
            macro_auroc_mean =('macro_auroc',  'mean'),
            macro_auroc_std  =('macro_auroc',  'std'),
            balanced_acc_mean=('balanced_acc', 'mean'),
            balanced_acc_std =('balanced_acc', 'std'),
            ece_mean         =('ece',          'mean'),
            ece_std          =('ece',          'std'),
            accuracy_mean    =('accuracy',     'mean'),
            accuracy_std     =('accuracy',     'std'),
        )
        .reset_index()
    )
    for col in summary.select_dtypes(include='float').columns:
        summary[col] = summary[col].round(6)
    summary = summary.sort_values(
        ['d', 'compressor'], ascending=[False, True]
    ).reset_index(drop=True)
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    os.makedirs(ARTIFACTS, exist_ok=True)

    print("=" * 65)
    print("CONFRONTO COMPRESSORI: B1 (PCA) vs B2 (VanillaAE) vs B3 (RegularizedAE)")
    print("Classificatore: LogisticRegression | Metrica principale: macro-F1")
    print("=" * 65 + "\n")

    # Valutazione dettagliata
    df = run_comparison()

    if df.empty:
        print("\n[WARNING] Nessun risultato — verificare che i CSV esistano.")
        return

    # Salvataggio dettaglio
    detail_path = f"{ARTIFACTS}/team_b_comparison.csv"
    df.to_csv(detail_path, index=False)

    # Riepilogo per (compressore, d)
    summary = make_summary(df)
    summary_path = f"{ARTIFACTS}/team_b_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("RIEPILOGO — medie su 3 seed per (compressore, d)")
    print("=" * 65)

    for metric, label in [
        ('macro_f1_mean',     'macro-F1'),
        ('macro_auroc_mean',  'macro-AUROC'),
        ('balanced_acc_mean', 'balanced accuracy'),
        ('ece_mean',          'ECE (↓ meglio)'),
    ]:
        pivot = summary.pivot(
            index='d', columns='compressor', values=metric
        ).sort_index(ascending=False)
        print(f"\n{label}:")
        print(pivot.to_string())

    print(f"\nDettaglio  → {detail_path}")
    print(f"Riepilogo  → {summary_path}")
    print("\n[DONE] Confronto completato.")


if __name__ == "__main__":
    main()