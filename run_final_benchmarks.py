import numpy as np
import pandas as pd
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# SVM con kernel RBF scala O(n²) sui campioni: su 87k campioni richiederebbe
# ore. 5.000 campioni stratificati sono sufficienti per una stima affidabile
# e riducono il tempo di fit da ore a pochi secondi per combinazione.
MAX_SVM_SAMPLES = 5_000


def load_cached_features(backbone, d, seed, split):
    path = f"artifacts/{backbone}/features/pca_d{d}_s{seed}_{split}.npz"
    data = np.load(path)
    return data['features'], data['labels'].ravel()


def train_all_classical(X_train, y_train, X_test, y_test, seed):
    """Addestra e valuta i 3 classificatori classici richiesti."""

    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=2000, random_state=seed)
    lr.fit(X_train, y_train)
    acc_lr = accuracy_score(y_test, lr.predict(X_test)) * 100

    # 2. MLP
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=seed)
    mlp.fit(X_train, y_train)
    acc_mlp = accuracy_score(y_test, mlp.predict(X_test)) * 100

    # 3. SVM RBF — subset stratificato per renderlo eseguibile in tempi ragionevoli.
    # Il fit avviene su MAX_SVM_SAMPLES campioni; la valutazione usa l'intero test set.
    if len(X_train) > MAX_SVM_SAMPLES:
        X_svm, _, y_svm, _ = train_test_split(
            X_train, y_train,
            train_size=MAX_SVM_SAMPLES,
            stratify=y_train,
            random_state=seed,
        )
    else:
        X_svm, y_svm = X_train, y_train

    svm = SVC(kernel='rbf', random_state=seed)
    svm.fit(X_svm, y_svm)
    acc_svm = accuracy_score(y_test, svm.predict(X_test)) * 100

    return acc_lr, acc_mlp, acc_svm


def run_single(d, seed, backbone):
    """Esegue i 3 classificatori per una coppia (d, seed). Ritorna dict con i risultati."""
    X_train, y_train = load_cached_features(backbone, d, seed, 'train')
    X_test,  y_test  = load_cached_features(backbone, d, seed, 'test')

    print(f"  >>> d={d} seed={seed} training su {len(X_train)} campioni "
          f"(SVM su max {MAX_SVM_SAMPLES})...", flush=True)

    acc_lr, acc_mlp, acc_svm = train_all_classical(X_train, y_train, X_test, y_test, seed)

    print(f"  [OK] d={d} seed={seed} LR={acc_lr:.2f}%  "
          f"MLP={acc_mlp:.2f}%  SVM={acc_svm:.2f}%", flush=True)

    return {"d": d, "seed": seed, "lr": acc_lr, "mlp": acc_mlp, "svm": acc_svm}


def main():
    start_total = time.time()
    dims     = [32, 16, 8, 4]
    seeds    = [11, 17, 29]
    backbone = 'pca'

    jobs = [(d, s) for d in dims for s in seeds]

    # Su Windows non superare i core fisici disponibili
    max_workers = min(len(jobs), os.cpu_count() or 1)

    print("=== START CLASSICAL SCREENING (FEATURE CACHE MODE) ===")
    print(f"SVM: fit su subset stratificato di {MAX_SVM_SAMPLES} campioni, "
          f"valutazione su test set completo.")
    print(f"[INFO] Avvio {len(jobs)} run su {max_workers} processi paralleli...\n")

    raw_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        
        futures = {
            executor.submit(run_single, d, s, backbone): (d, s)
            for d, s in jobs
        }
        for future in as_completed(futures):
            d, s = futures[future]
            try:
                raw_results.append(future.result())
            except FileNotFoundError:
                print(f"  [ERRORE] Feature cache per d={d} seed={s} non trovata. "
                      f"Esegui prima test.py!", flush=True)
            except Exception as e:
                print(f"  [ERRORE] d={d} seed={s} -> {e}", flush=True)

    # Aggrega per dimensione (mean e std sui 3 seed)
    df_raw = pd.DataFrame(raw_results)
    results = []
    for d in dims:
        subset = df_raw[df_raw['d'] == d]
        if subset.empty:
            continue
        results.append({
            'd':                    d,
            'PCA_LR_Ablation_mean': subset['lr'].mean(),
            'PCA_LR_Ablation_std':  subset['lr'].std(),
            'PCA_MLP_mean':         subset['mlp'].mean(),
            'PCA_MLP_std':          subset['mlp'].std(),
            'PCA_SVM_RBF_mean':     subset['svm'].mean(),
            'PCA_SVM_RBF_std':      subset['svm'].std(),
        })

    df = pd.DataFrame(results)
    os.makedirs("artifacts", exist_ok=True)
    df.to_csv("artifacts/final_benchmarks.csv", index=False)

    print("\n" + "=" * 55)
    print("TABELLA RIASSUNTIVA SCREENING CLASSICO")
    print("=" * 55)
    print(df.to_string(index=False))
    print(f"\nTempo totale: {(time.time() - start_total) / 60:.2f} minuti.")
    print("Salvato in: artifacts/final_benchmarks.csv")


# Obbligatorio su Windows: senza questo guard ogni worker
# rilancerebbe main() ricorsivamente (freeze_support error)
if __name__ == "__main__":
    main()