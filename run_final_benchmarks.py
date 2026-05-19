import numpy as np
import pandas as pd
import os
import time
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
    """
    Carica le feature salvate da test.py in formato .npz.
    Path coerente con test.py: artifacts/pca/features/pca_raw_d{d}_seed{seed}_{split}.npz
    """
    path = f"artifacts/{backbone}/features/pca_raw_d{d}_seed{seed}_{split}.npz"
    data = np.load(path)
    return data['features'], data['labels'].ravel()


def train_all_classical(X_train, y_train, X_test, y_test, seed):
    """Addestra e valuta i 3 classificatori classici richiesti."""

    # 1. Logistic Regression — regge l'intero training set
    lr = LogisticRegression(max_iter=2000, random_state=seed)
    lr.fit(X_train, y_train)
    acc_lr = accuracy_score(y_test, lr.predict(X_test)) * 100

    # 2. MLP — regge l'intero training set
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


def main():
    start_total = time.time()
    dims     = [32, 16, 8, 4]
    seeds    = [11, 17, 29]
    backbone = 'pca'
    results  = []

    print("=== START CLASSICAL SCREENING (FEATURE CACHE MODE) ===")
    print(f"SVM: fit su subset stratificato di {MAX_SVM_SAMPLES} campioni, "
          f"valutazione su test set completo.")

    for d in dims:
        print(f"\n>>> Analisi per d={d}")
        seed_results = {'lr': [], 'mlp': [], 'svm': []}

        for s in seeds:
            try:
                X_train, y_train = load_cached_features(backbone, d, s, 'train')
                X_test,  y_test  = load_cached_features(backbone, d, s, 'test')

                print(f"  Seed {s}: training su {len(X_train)} campioni "
                      f"(SVM su max {MAX_SVM_SAMPLES})...")
                acc_lr, acc_mlp, acc_svm = train_all_classical(
                    X_train, y_train, X_test, y_test, s
                )
                seed_results['lr'].append(acc_lr)
                seed_results['mlp'].append(acc_mlp)
                seed_results['svm'].append(acc_svm)

                print(f"  Seed {s}: LR={acc_lr:.2f}%  "
                      f"MLP={acc_mlp:.2f}%  SVM={acc_svm:.2f}%")

            except FileNotFoundError:
                print(f"  ERRORE: feature cache per d={d} seed={s} non trovata. "
                      f"Esegui prima test.py!")

        if seed_results['lr']:
            results.append({
                'd':                    d,
                'PCA_LR_Ablation_mean': np.mean(seed_results['lr']),
                'PCA_LR_Ablation_std':  np.std(seed_results['lr']),
                'PCA_MLP_mean':         np.mean(seed_results['mlp']),
                'PCA_MLP_std':          np.std(seed_results['mlp']),
                'PCA_SVM_RBF_mean':     np.mean(seed_results['svm']),
                'PCA_SVM_RBF_std':      np.std(seed_results['svm']),
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


if __name__ == "__main__":
    main()