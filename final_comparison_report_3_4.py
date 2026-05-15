import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_ablation_report():
    # Carica baseline classica (Week 3) e quantistica (Week 4)
    classical = pd.read_csv("artifacts/final_benchmarks.csv")
    quantum   = pd.read_csv("artifacts/week4_vqc_results.csv")

    # FIX H-1: la colonna 'ResNet_LR_Ablation' non esiste nel CSV prodotto da
    # run_final_benchmarks.py (corretto). Le colonne reali sono
    # 'PCA_LR_Ablation_mean' e 'PCA_LR_Ablation_std'.
    # Rinominiamo in modo esplicito per chiarezza nel report.
    if 'PCA_LR_Ablation_mean' not in classical.columns:
        raise KeyError(
            "Colonna 'PCA_LR_Ablation_mean' non trovata in final_benchmarks.csv. "
            "Verifica che run_final_benchmarks.py sia stato eseguito nella versione corretta."
        )

    classical_summary = classical[['d', 'PCA_LR_Ablation_mean']].rename(
        columns={'PCA_LR_Ablation_mean': 'Classical_LR_Acc'}
    )

    # FIX H-2: week4_vqc_results.csv ha una riga per ogni (d, seed), non per d.
    # Un merge diretto produce righe duplicate. Aggreghiamo prima per d
    # calcolando media e std della best accuracy tra i seed.
    quantum_summary = (
        quantum.groupby('d')['VQC_Best_Acc']
        .agg(VQC_Best_Acc_mean='mean', VQC_Best_Acc_std='std')
        .reset_index()
    )

    final_df = pd.merge(classical_summary, quantum_summary, on='d')

    print("\n=== REPORT FINALE: LINEAR COMPARATOR vs VQC ===")
    print(final_df.to_string(index=False))

    # Grafico di stabilizzazione
    os.makedirs("artifacts", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(final_df['d'], final_df['Classical_LR_Acc'],
             label='No-Quantum Ablation (Linear)', marker='o')
    plt.errorbar(final_df['d'], final_df['VQC_Best_Acc_mean'],
                 yerr=final_df['VQC_Best_Acc_std'],
                 label='VQC (RY + RealAmplitudes)', marker='s', capsize=4)
    plt.gca().invert_xaxis()
    plt.xlabel('Dimensione Latente (d)')
    plt.ylabel('Accuratezza (%)')
    plt.title('Stabilizzazione Baseline: Classico vs Quantistico')
    plt.legend()
    plt.grid(True)
    plt.savefig("artifacts/final_stabilization_plot.png")
    plt.close()
    print("Grafico salvato in artifacts/final_stabilization_plot.png")

if __name__ == "__main__":
    generate_ablation_report()