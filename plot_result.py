import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_plots(): # Funzione per generare i grafici di accuratezza durante il training del modello ibrido, utile per monitorare la convergenza e le prestazioni durante il training del modello ibrido
    log_file = "experiments/production_log.csv"
    if not os.path.exists(log_file):
        print("Errore: production_log.csv non trovato!")
        return

    df = pd.read_csv(log_file)
    
    # Grafico accuratezza per regime latente
    plt.figure(figsize=(10, 6))
    for d in df['d'].unique():
        subset = df[df['d'] == d]
        # Media tra i backbone per semplicità di visualizzazione
        avg_acc = subset.groupby('epoch')['val_acc'].mean()
        plt.plot(avg_acc.index, avg_acc.values, marker='o', label=f'Dimensione d={d}')
    
    plt.title('Convergenza Accuratezza VQC - Week 3') # Titolo del grafico per contestualizzare i risultati e monitorare la convergenza durante il training del modello ibrido
    plt.xlabel('Epoca')
    plt.ylabel('Validation Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiments/accuracy_trend.png')
    print("Grafico accuracy_trend.png generato in /experiments")

if __name__ == "__main__":
    generate_plots()