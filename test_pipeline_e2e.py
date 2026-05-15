import torch
from data_loader import get_data_loaders
from pca_res_compressors import ResNetCompressor
import numpy as np
from quantum_model import QuantumPipeline, min_max_scaler
from test import get_weights_path

# 1. Setup
flag = 'octmnist' # Dataset flag per ResNetCompressor, utile per garantire la coerenza con i dati utilizzati durante il training del modello ibrido
percorso_pesi = get_weights_path(flag)  # Utilizza il nuovo percorso
d = 32 # Dimensione latente (d=32, 16, 8, 4)
n_qubits = 4 # Numero di qubit per il circuito quantistico (4 qubit per 4 classi)
train_loader, _, _ = get_data_loaders(batch_size=128) # Carichiamo un batch di dati (batch_size=128 per testare la pipeline end-to-end su un singolo sample)
images, labels = next(iter(train_loader)) # Prendiamo il primo batch di immagini e label per testare la pipeline end-to-end, in modo da validare la corretta integrazione tra la parte classica (ResNet) e la parte quantistica (circuito parametrico) durante il forward pass del modello ibrido

# 2. Compressione (Backbone congelato)
compressor = ResNetCompressor(d_latent=d, data_flag=flag) # Inizializziamo il compressore ResNet con la dimensione latente specificata (d) per eseguire la compressione delle immagini e ottenere le feature da passare al circuito quantistico durante il forward pass del modello ibrido
u = compressor(images).detach().numpy()[0] # Prendiamo il primo sample

# 3. Scaling
u_scaled = min_max_scaler(u) # Applichiamo uno scaling min-max alle feature compresse per portarle nel range [0, pi], in modo da renderle adatte come input per i gate RY del circuito quantistico durante il forward pass del modello ibrido

# 4. Quantum Forward Pass (Mock)
qp = QuantumPipeline(n_qubits=n_qubits, d_latent=d) # Inizializziamo il pipeline quantistico con il numero di qubit e la dimensione latente specificati, in modo da costruire il circuito parametrico che sarà utilizzato per il forward pass del modello ibrido
qc = qp.build_circuit() # Costruiamo il circuito quantistico basato sulla configurazione specificata, in modo da prepararlo per l'esecuzione durante il forward pass del modello ibrido

# Assegniamo i valori reali al circuito
bound_qc = qc.assign_parameters({qp.features: u_scaled, qp.weights: np.random.randn(len(qp.weights))}) # Assegniamo i valori scalati delle feature compresse ai parametri di input del circuito quantistico e assegniamo valori casuali ai pesi del circuito (questi ultimi non sono rilevanti per questo test, in quanto stiamo validando solo la pipeline end-to-end e non l'efficacia del training dei pesi durante il forward pass del modello ibrido)

print(f"Feature compresse: {u}")
print(f"Feature scalate per Quantum: {u_scaled}")
print("\nCircuito finale pronto per l'esecuzione:")
print(bound_qc)

# 1. Setup
d = 16
n_qubits = 4
train_loader, _, _ = get_data_loaders(batch_size=128)
images, labels = next(iter(train_loader))

# 2. Compressione (Backbone congelato)
compressor = ResNetCompressor(d_latent=d, data_flag=flag)
u = compressor(images).detach().numpy()[0] # Prendiamo il primo sample

# 3. Scaling
u_scaled = min_max_scaler(u)

# 4. Quantum Forward Pass (Mock)
qp = QuantumPipeline(n_qubits=n_qubits, d_latent=d)
qc = qp.build_circuit()

# Assegniamo i valori reali al circuito
bound_qc = qc.assign_parameters({qp.features: u_scaled, qp.weights: np.random.randn(len(qp.weights))})

print(f"Feature compresse: {u}")
print(f"Feature scalate per Quantum: {u_scaled}")
print("\nCircuito finale pronto per l'esecuzione:")
print(bound_qc)

# 1. Setup
d = 8
n_qubits = 4
train_loader, _, _ = get_data_loaders(batch_size=128)
images, labels = next(iter(train_loader))

# 2. Compressione (Backbone congelato)
compressor = ResNetCompressor(d_latent=d, data_flag=flag)
u = compressor(images).detach().numpy()[0] # Prendiamo il primo sample

# 3. Scaling
u_scaled = min_max_scaler(u)

# 4. Quantum Forward Pass (Mock)
qp = QuantumPipeline(n_qubits=n_qubits, d_latent=d)
qc = qp.build_circuit()

# Assegniamo i valori reali al circuito
bound_qc = qc.assign_parameters({qp.features: u_scaled, qp.weights: np.random.randn(len(qp.weights))})

print(f"Feature compresse: {u}")
print(f"Feature scalate per Quantum: {u_scaled}")
print("\nCircuito finale pronto per l'esecuzione:")
print(bound_qc)

# 1. Setup
d = 4
n_qubits = 4
train_loader, _, _ = get_data_loaders(batch_size=128)
images, labels = next(iter(train_loader))

# 2. Compressione (Backbone congelato)
compressor = ResNetCompressor(d_latent=d, data_flag=flag)
u = compressor(images).detach().numpy()[0] # Prendiamo il primo sample

# 3. Scaling
u_scaled = min_max_scaler(u)

# 4. Quantum Forward Pass (Mock)
qp = QuantumPipeline(n_qubits=n_qubits, d_latent=d)
qc = qp.build_circuit()

# Assegniamo i valori reali al circuito
bound_qc = qc.assign_parameters({qp.features: u_scaled, qp.weights: np.random.randn(len(qp.weights))})

print(f"Feature compresse: {u}")
print(f"Feature scalate per Quantum: {u_scaled}")
print("\nCircuito finale pronto per l'esecuzione:")
print(bound_qc)