"""
week8_audit.py

Audit finale degli artifact Team B (Gruppo 22) prima del freeze della
Week 8 — verifica che tutti gli output attesi della pipeline esistano
e siano completi.

FIX rispetto alla versione precedente: controllava percorsi che non
corrispondono in alcun modo alla struttura reale del progetto — ogni
controllo avrebbe restituito MISSING per costruzione, indipendentemente
da quanto la pipeline fosse effettivamente completa. Nello specifico:

  - test_ids_{seed}.csv (uno per seed) invece di test_ids_fixed.csv
    (fisso, condiviso da tutti i seed) — esattamente il bug di data
    leakage/varianza spuria già trovato e corretto in data_loader.py.
  - models/compressor_{comp}_d{d}.pt — cartella e schema mai esistiti;
    i checkpoint reali sono in artifacts/sweep/{comp}_d{d}_s{seed}.pt,
    con seed (b2_b3_training.py salva un modello per ogni seed, non uno
    condiviso per d).
  - artifacts/features/pca_d{d}.pkl — B1 non salva un compressore
    serializzato; le componenti PCA sono già incorporate nei CSV.
  - ablation_linear_d{d}.csv (un file per d) invece di
    team_b_comparison.csv (un unico file aggregato con tutte le righe).
  - results_d{d}_s{s}_f{frac}.npz per-combinazione invece di
    fewshot_final_results.csv aggregato; seed=23 mai usato — confermato
    un refuso del documento di progetto dal docente (solo seed=11).

Riscritto per verificare i percorsi REALI prodotti da ciascuno script
della pipeline. Dove l'output è un CSV aggregato, l'audit verifica sia
l'esistenza del file sia la presenza di tutte le righe attese al suo
interno (join sulle chiavi compressore/d/seed), non solo l'esistenza
del file — un file presente ma con righe mancanti è comunque segnalato.
"""

import os

import pandas as pd


# ---------------------------------------------------------------------------
# Configurazione — deve rispecchiare esattamente le costanti usate nella
# pipeline reale (train_vqc_production.py, master_sweep_team_b.py, ecc.)
# ---------------------------------------------------------------------------
DIMS              = [32, 16, 8, 4]
SEEDS             = [11, 17, 29]
FEWSHOT_DIMS      = [8, 4]
FEWSHOT_SEEDS     = [11]   # seed=23 nel documento è un refuso confermato dal docente
FEWSHOT_FRACTIONS = [0.25, 0.10, 0.05]
COMPRESSORS       = ['B1', 'B2', 'B3']
SPLITS            = ['train', 'val', 'test']


# ---------------------------------------------------------------------------
# Helper di verifica
# ---------------------------------------------------------------------------
def check_file(category: str, item: str, path: str, report: list) -> bool:
    exists = os.path.exists(path)
    report.append({
        'Category': category, 'Item': item, 'Path': path,
        'Status': 'OK' if exists else 'MISSING',
    })
    return exists


def check_csv_rows(category: str, item: str, path: str, report: list,
                    expected_keys: pd.DataFrame, key_cols: list) -> bool:
    """
    Verifica che un CSV aggregato esista E contenga tutte le combinazioni
    attese (join sulle colonne chiave, es. compressor/d/seed).

    Args:
        expected_keys: DataFrame con tutte le combinazioni attese.
        key_cols:      colonne su cui fare il join (devono esistere nel CSV).
    """
    if not os.path.exists(path):
        report.append({'Category': category, 'Item': item, 'Path': path,
                        'Status': 'MISSING (file assente)'})
        return False
    try:
        df = pd.read_csv(path)
        missing_cols = [c for c in key_cols if c not in df.columns]
        if missing_cols:
            report.append({'Category': category, 'Item': item, 'Path': path,
                            'Status': f'ERRORE: colonne mancanti {missing_cols}'})
            return False

        merged = expected_keys.merge(df[key_cols].drop_duplicates(), on=key_cols, how='left', indicator=True)
        n_found   = (merged['_merge'] == 'both').sum()
        n_expected = len(expected_keys)

        status = 'OK' if n_found == n_expected else f'INCOMPLETO ({n_found}/{n_expected} combinazioni)'
        report.append({'Category': category, 'Item': item, 'Path': path, 'Status': status})
        return n_found == n_expected
    except Exception as e:
        report.append({'Category': category, 'Item': item, 'Path': path,
                        'Status': f'ERRORE LETTURA: {e}'})
        return False


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
def run_team_b_audit():
    report = []
    print("=== AUDIT FINALE — WEEK 8 — TEAM B (GRUPPO 22) ===\n")

    # 1. Split condivisi (generate_splits.py) ---------------------------------
    print("[1/8] Split condivisi...")
    for seed in SEEDS:
        check_file('Split condivisi', f'train_ids_{seed}',
                   f'dataset_splits/train_ids_{seed}.csv', report)
        check_file('Split condivisi', f'val_ids_{seed}',
                   f'dataset_splits/val_ids_{seed}.csv', report)
    # Test set FISSO — un solo file, condiviso da tutti i seed (non uno a testa)
    check_file('Split condivisi', 'test_ids_fixed',
               'dataset_splits/test_ids_fixed.csv', report)

    # 2. Feature raw ResNet18 (test.py) ----------------------------------------
    print("[2/8] Feature raw ResNet18...")
    for d in DIMS:
        for split in SPLITS:
            for seed in SEEDS:
                check_file('Feature raw (test.py)', f'd{d}_{split}_s{seed}',
                           f'artifacts/resnet/features/res_raw_d_{d}_{split}_s{seed}.npz',
                           report)

    # 3. Feature compresse B1/B2/B3 --------------------------------------------
    print("[3/8] Feature compresse B1/B2/B3...")
    for comp in COMPRESSORS:
        for d in DIMS:
            for split in SPLITS:
                for seed in SEEDS:
                    check_file(f'Feature compresse ({comp})', f'{comp}_d{d}_{split}_s{seed}',
                               f'artifacts/sweep/{comp}_pca_{split}_d{d}_seed{seed}.csv',
                               report)

    # 4. Checkpoint compressori B2/B3 (b2_b3_training.py) ----------------------
    # B1 non ha un checkpoint proprio: le componenti PCA sono incorporate
    # direttamente nei CSV del punto 3, non serializzate separatamente.
    print("[4/8] Checkpoint compressori B2/B3...")
    for comp in ['B2', 'B3']:
        for d in DIMS:
            for seed in SEEDS:
                check_file(f'Checkpoint ({comp})', f'{comp}_d{d}_s{seed}',
                           f'artifacts/sweep/{comp}_d{d}_s{seed}.pt', report)

    # 5. VQC — checkpoint e risultati aggregati (train_vqc_production.py) ------
    print("[5/8] VQC — checkpoint e risultati...")
    for comp in COMPRESSORS:
        for d in DIMS:
            for seed in SEEDS:
                check_file('VQC checkpoint', f'{comp}_d{d}_s{seed}',
                           f'experiments/models/best_vqc_{comp}_d{d}_seed{seed}.pth', report)

    vqc_keys = pd.DataFrame(
        [(c, d, s) for c in COMPRESSORS for d in DIMS for s in SEEDS],
        columns=['compressor', 'd', 'seed'],
    )
    check_csv_rows('VQC risultati aggregati', 'production_summary.csv',
                   'experiments/production_summary.csv', report,
                   vqc_keys, ['compressor', 'd', 'seed'])

    # 6. No-quantum ablation + screening (master_sweep_team_b.py) --------------
    print("[6/8] No-quantum ablation + screening...")
    check_csv_rows('No-quantum ablation (LR)', 'team_b_comparison.csv',
                   'artifacts/sweep/team_b_comparison.csv', report,
                   vqc_keys, ['compressor', 'd', 'seed'])
    check_file('No-quantum ablation (LR)', 'team_b_summary.csv',
               'artifacts/sweep/team_b_summary.csv', report)
    check_file('Screening comparatori', 'team_b_screening.csv',
               'artifacts/sweep/team_b_screening.csv', report)
    check_file('Screening comparatori', 'team_b_selected_comparator.csv',
               'artifacts/sweep/team_b_selected_comparator.csv', report)

    # 7. Baseline classica few-shot + sweep few-shot ----------------------------
    print("[7/8] Baseline classica few-shot + sweep few-shot...")
    check_csv_rows('Baseline classica few-shot', 'classical_summary.csv',
                   'experiments_classical/classical_summary.csv', report,
                   vqc_keys, ['compressor', 'd', 'seed'])

    fewshot_keys = pd.DataFrame(
        [(c, d, s, f) for c in COMPRESSORS for d in FEWSHOT_DIMS
         for s in FEWSHOT_SEEDS for f in FEWSHOT_FRACTIONS],
        columns=['compressor', 'd', 'seed', 'fraction'],
    )
    check_csv_rows('Sweep few-shot', 'fewshot_final_results.csv',
                   'artifacts/fewshot_final_results.csv', report,
                   fewshot_keys, ['compressor', 'd', 'seed', 'fraction'])

    # 8. Valutazione pulita + deliverable finale ---------------------------------
    print("[8/8] Valutazione pulita + deliverable finale...")
    check_file('Valutazione pulita', 'week8_evaluation_report.csv',
               'artifacts/week8_evaluation_report.csv', report)
    check_file('Deliverable finale', 'short_paper_team_b.tex',
               'short_paper_team_b.tex', report)
    check_file('Deliverable finale', 'short_paper_team_b.pdf',
               'short_paper_team_b.pdf', report)

    # --- Riepilogo -------------------------------------------------------------
    df_audit = pd.DataFrame(report)
    missing = df_audit[df_audit['Status'] != 'OK']

    print("\n--- Risultati Audit ---")
    print(f"Totale controlli: {len(df_audit)}")
    print(f"Elementi mancanti/incompleti: {len(missing)}")

    if len(missing) > 0:
        print("\nATTENZIONE: artifact mancanti o incompleti:")
        print(missing.to_string(index=False))
    else:
        print("\nSUCCESS: tutti gli artifact sono presenti e completi. Procedere al freeze.")

    os.makedirs("artifacts", exist_ok=True)
    out_path = "artifacts/audit_report_week8.csv"
    df_audit.to_csv(out_path, index=False)
    print(f"\nReport completo salvato in {out_path}")


if __name__ == "__main__":
    run_team_b_audit()