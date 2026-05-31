import os
import pandas as pd
import numpy as np

def run_team_b_audit():
    """
    Esegue l'audit finale degli artifact per il Team B (Gruppo 22)
    prima del freeze assoluto della Week 8.
    """

    REQUIRED_DIMS = [32, 16, 8, 4]
    REQUIRED_SEEDS_CLEAN = [11, 17, 29]
    REQUIRED_SEEDS_FEWSHOT = [11, 23]
    FEW_SHOT_FRACTIONS = [0.25, 0.10, 0.05]
    COMPRESSION_TYPES = ['B1', 'B2', 'B3']

    audit_report = []

    print("=== START FINAL AUDIT - WEEK 8 - TEAM B (GROUP 22) ===")

    shared_assets = [
        'dataset_splits/train_ids_11.csv',
        'dataset_splits/train_ids_17.csv',
        'dataset_splits/train_ids_29.csv',
        'dataset_splits/val_ids_11.csv',
        'dataset_splits/val_ids_17.csv',
        'dataset_splits/val_ids_29.csv',
        'dataset_splits/test_ids_11.csv',
        'dataset_splits/test_ids_17.csv',
        'dataset_splits/test_ids_29.csv',
        'artifacts/resnet/resnet_base_512_seed11_test.npz',
        'artifacts/resnet/resnet_base_512_seed17_test.npz',
        'artifacts/resnet/resnet_base_512_seed29_test.npz',
        'artifacts/resnet/resnet_base_512_seed11_val.npz',
        'artifacts/resnet/resnet_base_512_seed17_val.npz',
        'artifacts/resnet/resnet_base_512_seed29_val.npz',
    ]

    for asset in shared_assets:
        exists = os.path.exists(asset)
        audit_report.append({
            'Category': 'Shared Assets',
            'Item': asset,
            'Status': 'OK' if exists else 'MISSING'
        })

    # 3. Verifica Sweep Comune e Variazioni Team B
    for d in REQUIRED_DIMS:
        for comp in COMPRESSION_TYPES:
            model_path = (
                f"models/compressor_{comp}_d{d}.pt"
                if comp != 'B1'
                else f"artifacts/features/pca_d{d}.pkl"
            )
            exists = os.path.exists(model_path)
            audit_report.append({
                'Category': 'Compression Models',
                'Item': f"{comp}_d{d}",
                'Status': 'OK' if exists else 'MISSING'
            })

            for s in REQUIRED_SEEDS_CLEAN:
                res_path = f"artifacts/res_{comp}_d{d}_s{s}.npz"
                exists = os.path.exists(res_path)
                audit_report.append({
                    'Category': 'Clean Results',
                    'Item': f"{comp}_d{d}_s{s}",
                    'Status': 'OK' if exists else 'MISSING'
                })

    # 4. Verifica No-Quantum Ablation
    for d in REQUIRED_DIMS:
        ablation_path = f"artifacts/ablation_linear_d{d}.csv"
        exists = os.path.exists(ablation_path)
        audit_report.append({
            'Category': 'No-Quantum Ablation',
            'Item': f"d{d}",
            'Status': 'OK' if exists else 'MISSING'
        })

    # 5. Verifica Policy Few-Shot
    for f in FEW_SHOT_FRACTIONS:
        for s in REQUIRED_SEEDS_FEWSHOT:
            for d in REQUIRED_DIMS:
                fs_path = f"artifacts/fewshot/results_d{d}_s{s}_f{int(f*100)}.npz"
                exists = os.path.exists(fs_path)
                audit_report.append({
                    'Category': 'Few-Shot Results',
                    'Item': f"f{f}_s{s}_d{d}",
                    'Status': 'OK' if exists else 'MISSING'
                })

    # 6. Analisi e Salvataggio Report
    df_audit = pd.DataFrame(audit_report)
    missing_items = df_audit[df_audit['Status'] == 'MISSING']

    print("\n--- Risultati Audit ---")
    print(f"Totale controlli: {len(df_audit)}")
    print(f"Elementi mancanti: {len(missing_items)}")

    if len(missing_items) > 0:
        print("\nATTENZIONE: Mancano artifact critici per il freeze della Week 8!")
        print(missing_items.to_string(index=False))
    else:
        print("\nSUCCESS: Tutti gli artifact sono conformi. Procedere al freeze.")

    os.makedirs("artifacts", exist_ok=True)
    df_audit.to_csv("audit_report_week8.csv", index=False)


if __name__ == "__main__":
    run_team_b_audit()