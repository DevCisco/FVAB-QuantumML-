import torch
import torch.optim as optim
import numpy as np
import os
import pandas as pd
from stabilized_vqc_model import StabilizedVQC

def train_regime(d, s, backbone='resnet'): # Funzione per eseguire il training del modello VQC su un regime latente specifico (d) e backbone, utile per valutare le prestazioni del modello ibrido su diversi regimi latenti e backbone durante il training del modello ibrido
    # Caricamento feature cached (Week 3)
    train_data = np.load(f"artifacts/{backbone}/features/res_d{d}_seed{s}_train.npz") # Carichiamo le feature salvate in formato .npz per il backbone specificato e la dimensione latente, utile per definire i dati di input del modello ibrido durante il training del modello ibrido
    val_data = np.load(f"artifacts/{backbone}/features/res_d{d}_seed{s}_val.npz") # Carichiamo le feature salvate in formato .npz per il backbone specificato e la dimensione latente, utile per definire i dati di input del modello ibrido durante il training del modello ibrido
    
    X_train = torch.tensor(train_data['features']).float()[:5000] # Subsampling per velocità
    y_train = torch.tensor(train_data['labels']).long().squeeze()[:5000]
    X_val = torch.tensor(val_data['features']).float()[:1000]
    y_val = torch.tensor(val_data['labels']).long().squeeze()[:1000]

    model = StabilizedVQC(n_qubits=4, d_latent=d) # Inizializziamo il modello VQC stabilizzato con la dimensione latente specificata, utile per migliorare la stabilità e le prestazioni durante il training del modello ibrido
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Ottimizzatore Adam con learning rate più basso per fine-tuning, utile per garantire una buona convergenza durante il training del modello ibrido
    criterion = torch.nn.CrossEntropyLoss() # Loss function CrossEntropyLoss per il training del modello ibrido, adatta per problemi di classificazione multi-classe come quello affrontato con il dataset OCTMNIST durante il training del modello ibrido

    best_acc = 0 # Variabile per tenere traccia della migliore accuratezza ottenuta durante il training del modello ibrido, utile per implementare la logica di checkpointing e salvare i pesi del modello quando si ottiene un miglioramento durante il training del modello ibrido
    epochs = 5 # Numero di epoche per il training del modello ibrido, in modo da monitorare la convergenza e le prestazioni durante il training del modello ibrido

    for epoch in range(epochs): # Eseguiamo il numero di epoche specificato per il training del modello ibrido, in modo da monitorare la convergenza e le prestazioni durante il training del modello ibrido
        model.train() # Impostiamo il modello in modalità training, utile per abilitare il calcolo dei gradienti e l'aggiornamento dei pesi durante il training del modello ibrido
        optimizer.zero_grad() # Azzeriamo i gradienti prima del backward pass, utile per garantire che i gradienti vengano calcolati correttamente durante il training del modello ibrido
        output = model(X_train[:128]) # Batch di test
        loss = criterion(output, y_train[:128]) # Calcoliamo la loss per il batch di test, utile per monitorare la convergenza e le prestazioni durante il training del modello ibrido
        loss.backward() # Calcoliamo i gradienti, utile per aggiornare i pesi del modello durante il training del modello ibrido
        optimizer.step() # Aggiorniamo i pesi del modello, utile per migliorare le prestazioni e la convergenza durante il training del modello ibrido

        # Validation (Checkpoint logic)
        model.eval() # Impostiamo il modello in modalità evaluation, utile per disabilitare il calcolo dei gradienti e migliorare le prestazioni durante la valutazione del modello ibrido
        with torch.no_grad(): # Disabilitiamo il calcolo dei gradienti durante la valutazione del modello ibrido, in modo da risparmiare memoria e migliorare le prestazioni durante il training del modello ibrido
            val_out = model(X_val[:200]) # Otteniamo le predizioni del modello sul set di validazione, utile per calcolare l'accuratezza e monitorare le prestazioni durante il training del modello ibrido
            acc = (val_out.argmax(1) == y_val[:200]).float().mean().item() * 100 # Calcoliamo l'accuratezza sul set di validazione, utile per monitorare le prestazioni durante il training del modello ibrido
            
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), f"artifacts/checkpoints/vqc_d{d}_best.pth") # Salviamo i pesi del modello se otteniamo la migliore accuratezza, in modo da poterli utilizzare per la comparazione e la validazione della pipeline end-to-end durante il training del modello ibrido
        
        print(f"d={d} | Epoch {epoch+1} | Val Acc: {acc:.2f}%")
    
    return best_acc

if __name__ == "__main__":
    os.makedirs("artifacts/checkpoints", exist_ok=True) # Creiamo la directory per salvare i checkpoint se non esiste già, in modo da organizzare gli artifact generati durante il training del modello ibrido
    results = []
    for d in [32, 16, 8, 4]:
        for s in [11, 17, 29]:
            acc = train_regime(d, s)
            results.append({"d": d, "s": s, "VQC_Best_Acc": acc}) # Salviamo i risultati in una lista di dizionari, utile per creare un DataFrame e salvare i risultati in formato CSV per la comparazione e l'analisi durante il training del modello ibrido
    
    pd.DataFrame(results).to_csv("artifacts/week4_vqc_results.csv", index=False) # Salviamo i risultati in un file CSV, utile per la comparazione e l'analisi dei risultati durante il training del modello ibrido