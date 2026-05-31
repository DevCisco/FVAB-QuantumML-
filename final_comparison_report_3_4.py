import pandas as pd
import matplotlib.pyplot as plt
import os


def generate_ablation_report():
    classical = pd.read_csv("artifacts/final_benchmarks.csv")
    quantum   = pd.read_csv("artifacts/week4_vqc_results.csv")

    # Verifica che le colonne attese (prodotte da run_final_benchmarks.py
    # corretto) esistano davvero nel CSV prima di procedere.
    required_cols = {'d', 'PCA_LR_Ablation_mean', 'PCA_LR_Ablation_std'}
    missing = required_cols - set(classical.columns)
    if missing:
        raise KeyError(
            f"Colonne mancanti in final_benchmarks.csv: {missing}\n"
            f"Colonne presenti: {list(classical.columns)}\n"
            "Assicurati di aver eseguito la versione corretta di "
            "run_final_benchmarks.py che aggrega i risultati per seed."
        )

    classical_summary = classical[
        ['d', 'PCA_LR_Ablation_mean', 'PCA_LR_Ablation_std']
    ].rename(columns={
        'PCA_LR_Ablation_mean': 'Classical_LR_Acc_mean',
        'PCA_LR_Ablation_std':  'Classical_LR_Acc_std',
    })

    quantum_summary = (
        quantum.groupby('d')['VQC_Best_Acc']
        .agg(VQC_Best_Acc_mean='mean', VQC_Best_Acc_std='std')
        .reset_index()
    )

    final_df = pd.merge(classical_summary, quantum_summary, on='d')

    print("\n=== REPORT FINALE: LINEAR COMPARATOR vs VQC ===")
    print(final_df.to_string(index=False))

    # ── Grafico ──────────────────────────────────────────────────────
    os.makedirs("artifacts", exist_ok=True)
    plt.figure(figsize=(8, 5))

    plt.errorbar(
        final_df['d'], final_df['Classical_LR_Acc_mean'],
        yerr=final_df['Classical_LR_Acc_std'],
        label='No-Quantum Ablation (Linear)', marker='o', capsize=4,
    )
    plt.errorbar(
        final_df['d'], final_df['VQC_Best_Acc_mean'],
        yerr=final_df['VQC_Best_Acc_std'],
        label='VQC (RY + RealAmplitudes)', marker='s', capsize=4,
    )

    plt.gca().invert_xaxis()
    plt.xlabel('Dimensione Latente (d)')
    plt.ylabel('Accuratezza (%)')
    plt.title('Stabilizzazione Baseline: Classico vs Quantistico')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("artifacts/final_stabilization_plot.png", dpi=150)
    plt.close()
    print("Grafico salvato in artifacts/final_stabilization_plot.png")


if __name__ == "__main__":
    generate_ablation_report()