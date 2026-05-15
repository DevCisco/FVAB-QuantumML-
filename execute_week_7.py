import numpy as np
import torch
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fewshot_sampler import get_stratified_fraction_indices, save_fewshot_manifest
from vqc_fewshot_engine import train_vqc

# FIX BUG 10: tutto il codice eseguito a livello di modulo rendeva execute_week_7
# un modulo con effetti collaterali pesanti all'import (caricamento file, training,
# export CSV). Qualsiasi 'import execute_week_7' avviava involontariamente l'intero
# esperimento. Spostato tutto dentro main() protetta da __name__ == "__main__".

def main():
    SEEDS     = [11, 23]
    FRACTIONS = [0.25, 0.10, 0.05]
    DIMS      = [8, 4]
    RESULTS   = []

    for d in DIMS:
        train_data = np.load(f"artifacts/resnet/features/res_d{d}_train.npz")
        test_data  = np.load(f"artifacts/resnet/features/res_d{d}_test.npz")

        X_full = train_data['features']
        y_full = train_data['labels'].ravel()
        X_test = torch.tensor(test_data['features']).float()
        y_test = test_data['labels'].ravel()

        for s in SEEDS:
            for f in FRACTIONS:
                print(f"\n>>> RUN: d={d} | Seed={s} | Fraction={int(f*100)}%")

                # 1. Subset Stratificato
                idx = get_stratified_fraction_indices(y_full, f, s)
                save_fewshot_manifest(idx, f, s, d)

                X_train_fs = torch.tensor(X_full[idx]).float()
                y_train_fs = torch.tensor(y_full[idx]).long()

                # 2. Baseline: Logistic Regression
                # FIX BUG 11: random_state mancante → risultati non riproducibili
                # con solver stocastici (sag, saga). Aggiunto random_state=s.
                clf = LogisticRegression(C=0.5, max_iter=1000, random_state=s)
                clf.fit(X_train_fs.numpy(), y_train_fs.numpy())
                acc_lr = accuracy_score(y_test, clf.predict(X_test.numpy())) * 100

                # 3. VQC
                model = train_vqc(X_train_fs, y_train_fs, d)
                with torch.no_grad():
                    preds  = model(X_test[:400]).argmax(1)
                    acc_vqc = accuracy_score(y_test[:400], preds.numpy()) * 100

                print(f" ACC -> LR: {acc_lr:.2f}% | VQC: {acc_vqc:.2f}%")
                RESULTS.append({
                    "d": d, "seed": s, "fraction": f,
                    "acc_lr": acc_lr, "acc_vqc": acc_vqc
                })

    os.makedirs("artifacts", exist_ok=True)
    df = pd.DataFrame(RESULTS)
    df.to_csv("artifacts/fewshot_final_results.csv", index=False)
    print("\n[DONE] Tutti i test della Policy Few-shot sono stati completati.")


if __name__ == "__main__":
    main()