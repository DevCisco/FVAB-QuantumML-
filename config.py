import numpy as np

class Config:
    # --- Riproducibilità ---
    SEEDS = [11, 17, 29]

    # --- Dataset & Preprocessing ---
    # NOTA: OCTMNIST è nativo 28×28 a 1 canale (scala di grigi).
    # Con as_rgb=True in data_loader.py le immagini vengono convertite a 3 canali;
    # IMAGE_SIZE=224 viene applicato tramite transforms.Resize nella pipeline.
    DATASET_NAME = "octmnist"
    IMAGE_SIZE   = 224
    BATCH_SIZE   = 32

    # --- Regimi Latenti ---
    LATENT_DIMENSIONS = [32, 16, 8, 4]

    # --- Parametri Quantistici ---
    N_QUBITS      = 4
    ENCODING      = "RY"
    ANSATZ        = "RealAmplitudes"
    RE_UPLOADING  = True
    SCALING_RANGE = (0, 2 * np.pi)

    # --- Ottimizzazione ---
    LR                  = 0.001
    EPOCHS              = 50
    EARLY_STOP_PATIENCE = 7

    # --- Inizializzazione parametri circuitali ---
    INIT_RANGE = (-0.01, 0.01)