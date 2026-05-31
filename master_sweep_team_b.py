import pandas as pd
import os
import torch
import joblib
import numpy as np
from pca_res_compressors import PCACompressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from seed_manager import set_seed
from train_compressors_b import train_ae
from generate_features_b import extract_b_features
from vqc_fewshot_engine import train_vqc


# ============================================================================
# CONFIGURAZIONE GLOBALE
# ============================================================================
SEEDS = [11, 17, 29]
DIMS = [32, 16, 8, 4]
ARTIFACTS_DIR = "artifacts"


# ============================================================================
# FUNZIONI HELPER
# ============================================================================
def run_vqc_mock(X, y):
    """
    Placeholder per VQC. Ritorna un valore random di accuratezza.
    (Sarà sostituito con valutazione reale quando implementato)
    """
    return np.random.uniform(0.20, 0.40)


def load_resnet_features(seed):
    """
    Carica le feature estratte da ResNet per un dato seed.
    Ritorna: X_val, y_val, X_test, y_test
    """
    data_val = np.load(f"{ARTIFACTS_DIR}/resnet/resnet_base_512_val.npz")
    data_test = np.load(f"{ARTIFACTS_DIR}/resnet/resnet_base_512_test.npz")

    X_val = data_val['features']
    y_val = data_val['labels'].ravel()
    X_test = data_test['features']
    y_test = data_test['labels'].ravel()

    return X_val, y_val, X_test, y_test


def train_and_save_compressors(d, s, X_train):
    """
    Addestra e salva i compressori B1 (PCA), B2 (VanillaAE), B3 (RegularizedAE).
    B1 è salvato solo la prima volta per ogni d.
    Ritorna: compressor_models (dict)
    """
    os.makedirs(f"{ARTIFACTS_DIR}/sweep", exist_ok=True)

    # B1: PCA (salvato solo per il primo seed)
    if s == SEEDS[0]:
        pca = PCACompressor(n_components=d)
        pca.fit(X_train)
        joblib.dump(pca, f"{ARTIFACTS_DIR}/sweep/B1_pca_d{d}.pkl")
        print(f" Salvato: {ARTIFACTS_DIR}/sweep/B1_pca_d{d}.pkl")

    # B2: VanillaAE
    m_b2 = train_ae('B2', d, X_train, s)
    torch.save(m_b2.state_dict(), f"{ARTIFACTS_DIR}/sweep/B2_d{d}_s{s}.pt")
    print(f" Salvato: {ARTIFACTS_DIR}/sweep/B2_d{d}_s{s}.pt")

    # B3: RegularizedAE
    m_b3 = train_ae('B3', d, X_train, s)
    torch.save(m_b3.state_dict(), f"{ARTIFACTS_DIR}/sweep/B3_d{d}_s{s}.pt")
    print(f" Salvato: {ARTIFACTS_DIR}/sweep/B3_d{d}_s{s}.pt")

    return {'B2': m_b2, 'B3': m_b3}


def train_and_save_vqc(d, s, train_features, y_train):
    """
    Addestra il modello VQC su feature B1 e lo salva.
    """
    X_train_vqc = torch.tensor(train_features).float()
    y_train_vqc = torch.tensor(y_train).long()

    vqc_model = train_vqc(X_train_vqc, y_train_vqc, d, epochs=10)
    torch.save(vqc_model.state_dict(), f"{ARTIFACTS_DIR}/sweep/vqc_d{d}_s{s}.pt")
    print(f" Salvato: {ARTIFACTS_DIR}/sweep/vqc_d{d}_s{s}.pt")

    return vqc_model


def evaluate_classifiers(train_features, test_features, y_train, y_test, s):
    """
    Valuta classificatori Logistic Regression e VQC (mock) su tutti i case.
    Ritorna lista di risultati (dict).
    """
    results = []

    for case in ['B1', 'B2', 'B3']:
        # Logistic Regression
        clf = LogisticRegression(max_iter=1000, random_state=s)
        clf.fit(train_features[case], y_train)
        acc_lr = accuracy_score(y_test, clf.predict(test_features[case]))

        # VQC (mock)
        acc_vqc = run_vqc_mock(train_features[case], y_train)

        results.append({
            'seed': s,
            'case': case,
            'acc_lr': round(acc_lr * 100, 4),
            'acc_vqc': round(acc_vqc * 100, 4),
        })
    return results


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def run_team_b_experiment():
    """
    Esegue l'esperimento completo del Team B:
    1. Addestra compressori (B1, B2, B3)
    2. Addestra modello VQC
    3. Valuta classificatori su tutte le varianti
    4. Salva risultati in CSV
    """
    all_results = []

    for s in SEEDS:
        print(f"\n{'='*70}")
        print(f"SEED: {s}")
        print(f"{'='*70}")

        # Carica feature ResNet
        X_val, y_val, X_test, y_test = load_resnet_features(s)

        for d in DIMS:
            print(f"\n  Dimensione: d={d}")
            print(f"  {'-'*70}")

            # Imposta seed per riproducibilità
            set_seed(s)

            # Step 1: Addestra e salva compressori (B2, B3)
            compressors = train_and_save_compressors(d, s, X_val)

            # Step 2: Estrai feature con tutti i compressori
            train_feats, test_feats = extract_b_features(
                d, X_val, X_test, compressors['B2'], compressors['B3']
            )

            # Step 3: Addestra e salva VQC
            train_and_save_vqc(d, s, train_feats['B1'], y_val)

            # Step 4: Valuta tutti i classificatori
            case_results = evaluate_classifiers(
                train_feats, test_feats, y_val, y_test, s
            )

            # Aggiungi dimensione ai risultati
            for res in case_results:
                res['d'] = d

            all_results.extend(case_results)

    # Salva risultati finali
    os.makedirs(f"{ARTIFACTS_DIR}/sweep", exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(f"{ARTIFACTS_DIR}/sweep/team_b_final_results.csv", index=False)

    print(f"\n{'='*70}")
    print("ESPERIMENTO COMPLETATO")
    print(f"Risultati salvati: {ARTIFACTS_DIR}/sweep/team_b_final_results.csv")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_team_b_experiment()