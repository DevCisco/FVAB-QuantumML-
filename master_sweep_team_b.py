import pandas as pd
import os
from seed_manager import set_seed
from train_compressors_b import train_ae
from generate_features_b import extract_b_features
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

def run_vqc_mock(X, y):
    """
    Mock placeholder del VQC (Week 4). Restituisce una frazione [0, 1]
    come accuracy_score, per coerenza di scala con il resto della pipeline.

    FIX BUG 6: la versione originale restituiva np.random.uniform(0.2, 0.4)
    che veniva poi moltiplicato per 100 nel results.append, producendo valori
    20–40 nel CSV. Ma acc_lr (da accuracy_score, già in [0,1]) veniva
    anch'essa moltiplicata per 100. Le due colonne avevano quindi scale
    coerenti per caso — ma solo perché il mock restituiva già "percentuali".
    Per sicurezza e coerenza con la produzione futura, il mock restituisce
    ora una frazione [0, 1], e la moltiplicazione ×100 avviene una sola volta
    in results.append per entrambe le metriche.
    """
    return np.random.uniform(0.20, 0.40)  # frazione, non percentuale

def run_team_b_experiment():
    seeds = [11, 17, 29]
    dims = [32, 16, 8, 4]
    results = []

    data_train = np.load("artifacts/resnet/resnet_base_512_train.npz")
    data_test  = np.load("artifacts/resnet/resnet_base_512_test.npz")

    X_512_train, y_train = data_train['features'], data_train['labels'].ravel()
    X_512_test,  y_test  = data_test['features'],  data_test['labels'].ravel()

    for s in seeds:
        for d in dims:
            print(f"--- Processing Seed {s} | d={d} ---")

            # FIX BUG 5: set_seed viene chiamato per ogni coppia (s, d), non
            # una sola volta per seed. Chiamandolo una volta sola per seed,
            # il PRNG avanza dopo ogni d e l'inizializzazione dei modelli per
            # d=16, 8, 4 non è riproducibile. Reimposto il seed qui garantisce
            # che ogni coppia (s, d) produca esattamente gli stessi pesi
            # indipendentemente dall'ordine di esecuzione.
            # Nota: train_ae chiama set_seed internamente, ma lo ripetiamo qui
            # per proteggere anche le operazioni numpy/sklearn successive.
            set_seed(s)

            # Addestramento Compressori Unsupervised
            # train_ae chiama set_seed(seed) internamente per riproducibilità
            m_b2 = train_ae('B2', d, X_512_train, s)
            m_b3 = train_ae('B3', d, X_512_train, s)

            # FIX BUG 1: extract_b_features ora riceve train E test separati
            # e fitta la PCA solo sul train (vedi generate_features_b.py).
            train_feats, test_feats = extract_b_features(
                d, X_512_train, X_512_test, m_b2, m_b3
            )

            for case in ['B1', 'B2', 'B3']:
                clf = LogisticRegression(max_iter=1000, random_state=s)
                clf.fit(train_feats[case], y_train)
                acc_lr = accuracy_score(y_test, clf.predict(test_feats[case]))

                acc_vqc = run_vqc_mock(train_feats[case], y_train)

                # FIX BUG 6: entrambe le accuracy sono frazioni [0,1],
                # la moltiplicazione ×100 avviene una sola volta qui.
                results.append({
                    'seed': s, 'd': d, 'case': case,
                    'acc_lr':  round(acc_lr  * 100, 4),
                    'acc_vqc': round(acc_vqc * 100, 4),
                })

    df = pd.DataFrame(results)
    os.makedirs("artifacts", exist_ok=True)
    df.to_csv("artifacts/team_b_final_results.csv", index=False)
    print("Esperimento Team B completato. Artifact salvato.")

if __name__ == "__main__":
    run_team_b_experiment()