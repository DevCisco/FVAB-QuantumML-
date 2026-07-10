import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_results():
    df = pd.read_csv("artifacts/team_b_final_results.csv")

    # FIX BUG 7: groupby().mean() aggregava anche la colonna 'seed', producendo
    # seed=19.0 (media di 11, 17, 29) — un valore privo di senso come
    # identificatore. Aggreghiamo esplicitamente solo le colonne di metrica.
    df_avg = (
        df.groupby(['d', 'case'])[['acc_lr', 'acc_vqc']]
        .mean()
        .reset_index()
    )

    os.makedirs("artifacts", exist_ok=True)

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_avg, x='d', y='acc_lr', hue='case', marker='o')
    plt.title("Confronto Compressione Unsupervised (Ablazione Lineare)")
    plt.gca().invert_xaxis()
    plt.ylabel("Accuracy %")
    plt.grid(True)
    plt.savefig("artifacts/compression_comparison_lr.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_avg, x='d', y='acc_vqc', hue='case',
                 linestyle='--', marker='s')
    plt.title("Confronto Compressione Unsupervised (VQC)")
    plt.gca().invert_xaxis()
    plt.ylabel("Accuracy %")
    plt.grid(True)
    plt.savefig("artifacts/compression_comparison_vqc.png")
    plt.close()

    print("Grafici salvati in artifacts/")

if __name__ == "__main__":
    plot_results()