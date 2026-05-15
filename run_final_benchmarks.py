import numpy as np
import pandas as pd
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# ECE — implementazione esplicita, protocollo fissato dal bando §25-28
# ---------------------------------------------------------------------------
# Protocollo:
#   • Top-label multiclass calibration error
#   • 15 bin uniformi su [0, 1]  →  edges = np.linspace(0, 1, 16)
#   • Norma L1: ECE = Σ_b (|b| / N) * |acc(b) − conf(b)|
#
# "Top-label" = per ogni campione si usa solo la coppia
#   (max_k p_k,  1[argmax_k p_k == y])
# Non si usa la probabilità della classe vera (quella sarebbe la classwise ECE).
# Ultimo bin chiuso a destra per includere confidenza = 1.0 esatta.

ECE_N_BINS = 15


def compute_ece(y_true: np.ndarray, proba: np.ndarray) -> float:
    """
    Expected Calibration Error — top-label, 15 bin uniformi, norma L1.

    Parametri
    ---------
    y_true : (N,)   — label intere ground-truth
    proba  : (N, C) — probabilità da predict_proba

    Restituisce
    -----------
    ece : float in [0, 1]
    """
    n = len(y_true)
    confidences = proba.max(axis=1)           # prob. massima per campione
    predictions = proba.argmax(axis=1)        # classe predetta
    correct     = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, ECE_N_BINS + 1)
    ece = 0.0

    for k in range(ECE_N_BINS):
        lo, hi = bin_edges[k], bin_edges[k + 1]
        # Ultimo bin chiuso a destra per catturare confidenza == 1.0
        mask = (confidences >= lo) & (confidences < hi) if k < ECE_N_BINS - 1 \
               else (confidences >= lo) & (confidences <= hi)

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        acc_bin  = correct[mask].mean()
        conf_bin = confidences[mask].mean()
        ece += (n_bin / n) * abs(acc_bin - conf_bin)

    return float(ece)


# ---------------------------------------------------------------------------
# I/O feature
# ---------------------------------------------------------------------------

def load_cached_features(backbone, d, seed, split):
    path = f"artifacts/{backbone}/features/pca_d{d}_seed{seed}_{split}.npz"
    data = np.load(path)
    return data['features'], data['labels'].ravel()


# ---------------------------------------------------------------------------
# Calcolo metrico unificato
# ---------------------------------------------------------------------------

def _compute_metrics(y_true, y_pred, y_proba) -> dict:
    """
    Calcola le quattro metriche di protocollo.

    1. Macro-AUROC  — roc_auc_score, multi_class='ovr', average='macro'
                      Input: probabilità di classe (predict_proba), non argmax.
    2. Macro-F1     — f1_score, average='macro', zero_division=0
                      Input: label argmax. zero_division=0 evita NaN su classi
                      assenti nel test set (scenari few-shot/sbilanciati).
    3. Balanced Acc — balanced_accuracy_score (= macro recall)
                      Input: label argmax.
    4. ECE          — top-label, 15 bin uniformi, L1  (vedi compute_ece)
                      Input: probabilità di classe.

    Tutti i valori restituiti sono in [0, 1] (non moltiplicati per 100).
    """
    auroc = roc_auc_score(
        y_true, y_proba,
        multi_class='ovr',   # one-vs-rest per ogni classe
        average='macro'      # media non pesata tra le C aree
    )
    macro_f1 = f1_score(
        y_true, y_pred,
        average='macro',
        zero_division=0
    )
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    ece     = compute_ece(y_true, y_proba)

    return {'auroc': auroc, 'macro_f1': macro_f1, 'balanced_acc': bal_acc, 'ece': ece}


# ---------------------------------------------------------------------------
# Training dei tre modelli classici
# ---------------------------------------------------------------------------

def train_all_classical(X_train, y_train, X_test, y_test, seed):
    """
    Addestra LR, MLP, SVM-RBF e calcola le quattro metriche per ciascuno.

    Nota su SVC: istanziato con probability=True (Platt scaling) per abilitare
    predict_proba. Senza questo flag roc_auc_score e compute_ece non possono
    ottenere le probabilità di classe e crasherebbero con AttributeError.
    """
    # Logistic Regression
    lr = LogisticRegression(max_iter=2000, random_state=seed)
    lr.fit(X_train, y_train)
    m_lr = _compute_metrics(y_test, lr.predict(X_test), lr.predict_proba(X_test))

    # MLP (1 hidden layer, 64 nodi)
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=seed)
    mlp.fit(X_train, y_train)
    m_mlp = _compute_metrics(y_test, mlp.predict(X_test), mlp.predict_proba(X_test))

    # RBF-SVM — probability=True obbligatorio per predict_proba
    svm = SVC(kernel='rbf', probability=True, random_state=seed)
    svm.fit(X_train, y_train)
    m_svm = _compute_metrics(y_test, svm.predict(X_test), svm.predict_proba(X_test))

    return m_lr, m_mlp, m_svm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODELS  = ['LR', 'MLP', 'SVM_RBF']
METRICS = ['auroc', 'macro_f1', 'balanced_acc', 'ece']


def main():
    start_total = time.time()
    dims     = [32, 16, 8, 4]
    seeds    = [11, 17, 29]
    backbone = 'pca'
    prefix   = 'PCA'
    results  = []

    print("=== START CLASSICAL SCREENING (FEATURE CACHE MODE) ===")
    print(
        f"Protocollo metriche fissato:\n"
        f"  • Macro-AUROC  : OvR sulle probabilità di classe, average='macro'\n"
        f"  • Macro-F1     : argmax, average='macro', zero_division=0\n"
        f"  • Balanced Acc : argmax (= macro recall)\n"
        f"  • ECE          : top-label, {ECE_N_BINS} bin uniformi, norma L1\n"
    )

    for d in dims:
        print(f"\n>>> Analisi per d={d}")
        row = {'d': d}

        # Accumulatori: dict[modello][metrica] → lista di valori sui seed
        acc = {m: {k: [] for k in METRICS} for m in MODELS}

        for s in seeds:
            print(f"    Caricamento feature {backbone.upper()} (seed={s})...")
            try:
                X_train, y_train = load_cached_features(backbone, d, s, 'train')
                X_test,  y_test  = load_cached_features(backbone, d, s, 'test')

                print(f"    Training (N_train={len(X_train)})...")
                m_lr, m_mlp, m_svm = train_all_classical(
                    X_train, y_train, X_test, y_test, s
                )

                for model_name, metrics in zip(MODELS, [m_lr, m_mlp, m_svm]):
                    for k, v in metrics.items():
                        acc[model_name][k].append(v)

                print(
                    f"    seed={s} | "
                    f"LR  auroc={m_lr['auroc']:.3f} f1={m_lr['macro_f1']:.3f} "
                    f"bacc={m_lr['balanced_acc']:.3f} ece={m_lr['ece']:.4f} | "
                    f"MLP auroc={m_mlp['auroc']:.3f} f1={m_mlp['macro_f1']:.3f} "
                    f"bacc={m_mlp['balanced_acc']:.3f} ece={m_mlp['ece']:.4f} | "
                    f"SVM auroc={m_svm['auroc']:.3f} f1={m_svm['macro_f1']:.3f} "
                    f"bacc={m_svm['balanced_acc']:.3f} ece={m_svm['ece']:.4f}"
                )

            except FileNotFoundError:
                print(
                    f"    ERRORE: cache {backbone} d={d} seed={s} non trovata. "
                    f"Esegui prima il modulo di feature extraction."
                )

        # Salva media ± std su tutti i seed disponibili
        for model_name in MODELS:
            for k in METRICS:
                vals = acc[model_name][k]
                if vals:
                    col = f"{prefix}_{model_name}_{k}"
                    row[f"{col}_mean"] = round(float(np.mean(vals)), 4)
                    row[f"{col}_std"]  = round(float(np.std(vals)),  4)

        results.append(row)

    df = pd.DataFrame(results)
    os.makedirs("artifacts", exist_ok=True)
    df.to_csv("artifacts/final_benchmarks.csv", index=False)

    print("\n" + "=" * 60)
    print("TABELLA RIASSUNTIVA SCREENING CLASSICO")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\nTempo totale: {(time.time() - start_total) / 60:.2f} minuti.")


if __name__ == "__main__":
    main()