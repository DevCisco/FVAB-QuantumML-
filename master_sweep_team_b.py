"""
master_sweep_team_b.py

Confronto sistematico dei tre compressori B1/B2/B3 con il protocollo di
screening classico richiesto dal documento di progetto: LogisticRegression,
MLP (un hidden layer), RBF-SVM — con selezione formale del comparatore
primario su solo validation set, non modificabile dopo la selezione.

Prerequisiti — questi file devono esistere prima di eseguire questo script:
  • B1: artifacts/sweep/B1_pca_{split}_d{d}_seed{seed}.csv  → test.py
  • B2: artifacts/sweep/B2_pca_{split}_d{d}_seed{seed}.csv  → b2_b3_training.py
  • B3: artifacts/sweep/B3_pca_{split}_d{d}_seed{seed}.csv  → b2_b3_training.py

Protocollo (documento di progetto, sezione "Confronto classico"):
  "Stesso screening per tutti: logistic regression, MLP a un hidden layer,
  RBF-SVM. La logistic regression coincide con la no-quantum ablation
  obbligatoria su feature compresse."
  "Selezione del comparatore classico: SOLO validation set, metrica
  macro-F1, media sui seed 11,17,29. Il modello selezionato viene
  congelato dopo Week 4 e non può essere cambiato."

Output:
  • artifacts/sweep/team_b_screening.csv   — dettaglio COMPLETO: i 3 modelli,
      val+test, per ogni (compressore, d, seed) — 108 righe totali.
  • artifacts/sweep/team_b_comparison.csv  — dettaglio SOLO del comparatore
      selezionato (metriche di test), stesso schema di colonne delle
      versioni precedenti di questo script — compatibilità con il paper.
  • artifacts/sweep/team_b_summary.csv     — media/std per (compressore, d)
      del comparatore selezionato.

Il VQC non è incluso qui — è gestito da train_vqc_production.py.
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
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
MODEL_TYPES  = ['LR', 'MLP', 'RBFSVM']
ARTIFACTS    = "artifacts/sweep"

# Path CSV per ogni compressore — devono corrispondere esattamente a quelli
# prodotti da test.py (B1) e b2_b3_training.py (B2, B3).
CSV_PATHS = {
    'B1': "artifacts/sweep/B1_pca_{split}_d{d}_seed{seed}.csv",
    'B2': "artifacts/sweep/B2_pca_{split}_d{d}_seed{seed}.csv",
    'B3': "artifacts/sweep/B3_pca_{split}_d{d}_seed{seed}.csv",
}

# Sottocampionamento stratificato per il fit di RBF-SVM. Il training pool
# completo (~87.729 immagini per seed) renderebbe il fit di un kernel RBF
# con probability=True proibitivamente lento (complessità O(n^2)-O(n^3) in
# scikit-learn per SVC). Sottocampionamento deterministico, seed-dipendente
# — pratica comune e documentata per metodi kernel su dataset di grandi
# dimensioni. Non applicato a LR/MLP, che scalano bene sul train completo.
RBFSVM_TRAIN_SUBSAMPLE = 5000


# ---------------------------------------------------------------------------
# Lettura CSV
# ---------------------------------------------------------------------------
def load_features(compressor: str, split: str, d: int, seed: int) -> tuple:
    """
    Carica feature e label dal CSV del compressore specificato.

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


def subsample_stratified(X: np.ndarray, y: np.ndarray, n_samples: int, seed: int) -> tuple:
    """
    Sottocampionamento stratificato deterministico — usato solo per il fit
    di RBF-SVM (vedi RBFSVM_TRAIN_SUBSAMPLE). Se X ha già <= n_samples righe,
    restituisce X, y invariati.
    """
    if len(X) <= n_samples:
        return X, y
    rng          = np.random.default_rng(seed)
    classes      = np.unique(y)
    per_class_n  = n_samples // len(classes)
    chosen_idx   = []
    for c in classes:
        c_idx   = np.where(y == c)[0]
        chosen  = rng.choice(c_idx, size=min(per_class_n, len(c_idx)), replace=False)
        chosen_idx.extend(chosen.tolist())
    chosen_idx = np.array(chosen_idx)
    return X[chosen_idx], y[chosen_idx]


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------
def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 15) -> float:
    """
    ECE per classificazione multiclasse con binning uniforme (top-1
    confidence, norma L1) — 15 bin richiesti dal documento di progetto.
    """
    confidences = y_proba.max(axis=1)
    predictions = y_proba.argmax(axis=1)
    correct     = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
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
# Costruzione classificatore
# ---------------------------------------------------------------------------
def build_classifier(model_type: str, seed: int):
    """
    Istanzia il classificatore richiesto dal protocollo di screening.

    'LR'     — Logistic Regression multinomiale (coincide con la
               no-quantum ablation obbligatoria).
    'MLP'    — MLP a un hidden layer (64 unità, early stopping per
               limitare il tempo di training).
    'RBFSVM' — SVM con kernel RBF, probability=True per AUROC/ECE
               (fit su un sottocampione, vedi RBFSVM_TRAIN_SUBSAMPLE).
    """
    if model_type == 'LR':
        return LogisticRegression(
            max_iter=1000, random_state=seed,
            solver='lbfgs'
        )
    elif model_type == 'MLP':
        return MLPClassifier(
            hidden_layer_sizes=(64,), max_iter=300, random_state=seed,
            early_stopping=True, n_iter_no_change=10,
        )
    elif model_type == 'RBFSVM':
        return SVC(kernel='rbf', probability=True, random_state=seed)
    else:
        raise ValueError(f"model_type sconosciuto: {model_type}")


# ---------------------------------------------------------------------------
# Valutazione — un modello, un compressore, una tripla (d, seed)
# ---------------------------------------------------------------------------
def evaluate_compressor(compressor: str, d: int, seed: int, model_type: str) -> dict:
    """
    Addestra il classificatore richiesto su train e valuta su VAL e TEST
    con quattro metriche: macro-F1, macro-AUROC (OvR), balanced accuracy, ECE.

    Il val è necessario per lo step di selezione del comparatore primario
    (solo validation, per protocollo); il test per il risultato finale
    riportato nel paper.
    """
    X_train, y_train = load_features(compressor, 'train', d, seed)
    X_val,   y_val   = load_features(compressor, 'val',   d, seed)
    X_test,  y_test  = load_features(compressor, 'test',  d, seed)

    X_fit, y_fit = X_train, y_train
    if model_type == 'RBFSVM':
        X_fit, y_fit = subsample_stratified(X_train, y_train, RBFSVM_TRAIN_SUBSAMPLE, seed)

    clf = build_classifier(model_type, seed)
    clf.fit(X_fit, y_fit)

    def _metrics(X, y):
        y_pred  = clf.predict(X)
        y_proba = clf.predict_proba(X)
        return {
            'macro_f1':     float(f1_score(y, y_pred, average='macro', zero_division=0)),
            'macro_auroc':  float(roc_auc_score(y, y_proba, multi_class='ovr', average='macro')),
            'balanced_acc': float(balanced_accuracy_score(y, y_pred)),
            'ece':          compute_ece(y, y_proba, n_bins=15),
            'accuracy':     float(accuracy_score(y, y_pred)),
        }

    val_m  = _metrics(X_val,  y_val)
    test_m = _metrics(X_test, y_test)

    row = {'model_type': model_type, 'compressor': compressor, 'd': d, 'seed': seed}
    for k, v in val_m.items():
        row[f'val_{k}'] = round(v, 6)
    for k, v in test_m.items():
        row[f'test_{k}'] = round(v, 6)
    return row


# ---------------------------------------------------------------------------
# Screening completo — tutti i modelli, tutte le combinazioni
# ---------------------------------------------------------------------------
def run_screening() -> pd.DataFrame:
    """
    Esegue la valutazione per tutti i (model_type, compressore, d, seed) —
    3 x 3 x 4 x 3 = 108 combinazioni totali.
    """
    results = []
    jobs    = [
        (mt, c, d, s)
        for mt in MODEL_TYPES for c in COMPRESSORS for d in DIMS for s in SEEDS
    ]
    n_total = len(jobs)

    for i, (model_type, compressor, d, seed) in enumerate(jobs, 1):
        try:
            row = evaluate_compressor(compressor, d, seed, model_type)
            results.append(row)
            print(
                f"[{i:3d}/{n_total}] {model_type:7s} {compressor} d={d:2d} seed={seed} | "
                f"val F1: {row['val_macro_f1']:.4f} | test F1: {row['test_macro_f1']:.4f}",
                flush=True,
            )
        except FileNotFoundError as e:
            print(f"[SKIP] {model_type} {compressor} d={d} seed={seed} → {e}", flush=True)
        except Exception as e:
            print(f"[ERROR] {model_type} {compressor} d={d} seed={seed} → {e}", flush=True)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Selezione del comparatore primario — SOLO validation, media su tutto
# ---------------------------------------------------------------------------
def select_primary_comparator(screening_df: pd.DataFrame) -> str:
    """
    Seleziona il comparatore classico primario secondo il protocollo del
    documento di progetto: SOLO validation set, metrica macro-F1, media su
    tutte le combinazioni (compressore, d) e sui 3 seed. Il modello con la
    media più alta viene selezionato e congelato da questo punto in poi.
    """
    means = screening_df.groupby('model_type')['val_macro_f1'].mean().sort_values(ascending=False)

    print("\n" + "=" * 65)
    print("SELEZIONE COMPARATORE PRIMARIO — solo validation, media su tutte le combinazioni")
    print("=" * 65)
    print(means.round(6).to_string())

    selected = means.index[0]
    print(f"\n>>> Comparatore selezionato: {selected} <<<")
    print("Congelato da questo punto in poi (documento di progetto, Week 4).")
    return selected


# ---------------------------------------------------------------------------
# Riepilogo per (compressore, d) — solo comparatore selezionato, metriche test
# ---------------------------------------------------------------------------
def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola media e std delle metriche di TEST sui 3 seed, per ogni
    coppia (compressore, d), sul solo comparatore selezionato.
    """
    summary = (
        df.groupby(['compressor', 'd'])
        .agg(
            macro_f1_mean    =('test_macro_f1',     'mean'),
            macro_f1_std     =('test_macro_f1',     'std'),
            macro_auroc_mean =('test_macro_auroc',  'mean'),
            macro_auroc_std  =('test_macro_auroc',  'std'),
            balanced_acc_mean=('test_balanced_acc', 'mean'),
            balanced_acc_std =('test_balanced_acc', 'std'),
            ece_mean         =('test_ece',          'mean'),
            ece_std          =('test_ece',          'std'),
            accuracy_mean    =('test_accuracy',     'mean'),
            accuracy_std     =('test_accuracy',     'std'),
        )
        .reset_index()
    )
    for col in summary.select_dtypes(include='float').columns:
        summary[col] = summary[col].round(6)
    summary = summary.sort_values(['d', 'compressor'], ascending=[False, True]).reset_index(drop=True)
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    os.makedirs(ARTIFACTS, exist_ok=True)

    print("=" * 65)
    print("SCREENING COMPARATORI CLASSICI: LogisticRegression, MLP, RBF-SVM")
    print(f"Compressori: B1 (PCA) / B2 (VanillaAE) / B3 (RegularizedAE)")
    print("=" * 65 + "\n")

    # — Screening completo: 108 combinazioni (3 modelli x 3 compressori x 4 d x 3 seed)
    screening_df = run_screening()

    if screening_df.empty:
        print("\n[WARNING] Nessun risultato — verificare che i CSV esistano.")
        return

    screening_path = f"{ARTIFACTS}/team_b_screening.csv"
    screening_df.to_csv(screening_path, index=False)

    # — Selezione del comparatore primario (solo validation, per protocollo).
    # FIX: la selezione è un'informazione AGGIUNTIVA — riportata in un file
    # separato — e non sostituisce più la no-quantum ablation obbligatoria.
    # Il documento distingue esplicitamente le due cose: "la logistic
    # regression coincide con la no-quantum ablation obbligatoria" (fissa,
    # sempre LR), mentre "il comparatore selezionato" (screening a 3 vie)
    # è un riferimento ulteriore, che può risultare in un modello diverso
    # da LR (qui: MLP). Confonderle porterebbe silenziosamente a riportare
    # in Tabella 1 un modello diverso da quello richiesto dal documento.
    selected = select_primary_comparator(screening_df)

    selected_df = screening_df[screening_df['model_type'] == selected].copy()
    selected_summary_rows = []
    for (compressor, d), group in selected_df.groupby(['compressor', 'd']):
        selected_summary_rows.append({
            'compressor': compressor, 'd': d,
            'val_macro_f1_mean':  round(group['val_macro_f1'].mean(), 6),
            'test_macro_f1_mean': round(group['test_macro_f1'].mean(), 6),
        })
    selected_path = f"{ARTIFACTS}/team_b_selected_comparator.csv"
    pd.DataFrame(selected_summary_rows).sort_values(
        ['d', 'compressor'], ascending=[False, True]
    ).to_csv(selected_path, index=False)

    # — No-quantum ablation obbligatoria: SEMPRE LogisticRegression,
    #   indipendentemente dall'esito della selezione sopra.
    df = screening_df[screening_df['model_type'] == 'LR'].copy()
    df = df.rename(columns={
        'test_macro_f1':     'macro_f1',
        'test_macro_auroc':  'macro_auroc',
        'test_balanced_acc': 'balanced_acc',
        'test_ece':          'ece',
        'test_accuracy':     'accuracy',
    })[['compressor', 'd', 'seed', 'macro_f1', 'macro_auroc', 'balanced_acc', 'ece', 'accuracy']]

    detail_path = f"{ARTIFACTS}/team_b_comparison.csv"
    df.to_csv(detail_path, index=False)

    summary = make_summary(
        screening_df[screening_df['model_type'] == 'LR'].copy()
    )
    summary_path = f"{ARTIFACTS}/team_b_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("RIEPILOGO — no-quantum ablation obbligatoria (Logistic Regression), medie su 3 seed")
    print("=" * 65)

    for metric, label in [
        ('macro_f1_mean',     'macro-F1'),
        ('macro_auroc_mean',  'macro-AUROC'),
        ('balanced_acc_mean', 'balanced accuracy'),
        ('ece_mean',          'ECE (↓ meglio)'),
    ]:
        pivot = summary.pivot(index='d', columns='compressor', values=metric).sort_index(ascending=False)
        print(f"\n{label}:")
        print(pivot.to_string())

    print(f"\nScreening completo (3 modelli) → {screening_path}")
    print(f"Comparatore selezionato ({selected}), riferimento aggiuntivo → {selected_path}")
    print(f"No-quantum ablation (LR, obbligatoria) → {detail_path}")
    print(f"Riepilogo ablation (LR) → {summary_path}")
    print("\n[DONE] Screening e selezione completati.")


if __name__ == "__main__":
    main()