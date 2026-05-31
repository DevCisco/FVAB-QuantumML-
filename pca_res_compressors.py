import numpy as np
from sklearn.decomposition import IncrementalPCA
import pickle
import torch.nn as nn
from torchvision import models
import os
import torch
from medmnist import INFO
import zipfile
import requests


class PCACompressor:
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = IncrementalPCA(n_components=n_components)
        self.is_fitted = False

    def fit(self, data_loader, batch_size=256):
        print(f"--- Fitting PCA per d={self.n_components} ---")
        
        # Gestisci sia array numpy che DataLoader
        if isinstance(data_loader, np.ndarray) or isinstance(data_loader, torch.Tensor):
            # Input è un array numpy o tensor
            if isinstance(data_loader, torch.Tensor):
                data_array = data_loader.numpy()
            else:
                data_array = data_loader
            
            # Reshape se necessario
            if data_array.ndim > 2:
                data_array = data_array.reshape(data_array.shape[0], -1)
            
            # Fit in batch
            n_samples = data_array.shape[0]
            for i in range(0, n_samples, batch_size):
                batch = data_array[i:i + batch_size]
                self.pca.partial_fit(batch)
                if (i + batch_size) % (batch_size * 50) == 0:
                    print(f"   [Batch {i + batch_size}] Processato.")
        else:
            # Input è un DataLoader
            for i, (images, _) in enumerate(data_loader):
                x = images.numpy().reshape(images.shape[0], -1)
                self.pca.partial_fit(x)
                if (i + 1) % 50 == 0:
                    print(f"   [Batch {i+1}] Processato.")
        
        self.is_fitted = True

    def transform(self, batch_images):
        if not self.is_fitted:
            raise RuntimeError("La PCA deve essere fittata prima di chiamare transform()!")
        if isinstance(batch_images, torch.Tensor):
            batch_images = batch_images.numpy()
        x = batch_images.reshape(batch_images.shape[0], -1)
        return self.pca.transform(x)

    def fit_transform(self, data_array, batch_size=256):
        """Fit e trasforma su un array numpy già pronto (non un DataLoader).
        Processa in batch per evitare memory overflow su array grandi."""
        if isinstance(data_array, torch.Tensor):
            data_array = data_array.numpy()
        # Reshape 4D array (batch, channels, height, width) to 2D (batch, features)
        if data_array.ndim > 2:
            data_array = data_array.reshape(data_array.shape[0], -1)
        
        # Fit in batch per gestire dataset grandi
        n_samples = data_array.shape[0]
        for i in range(0, n_samples, batch_size):
            batch = data_array[i:i + batch_size]
            self.pca.partial_fit(batch)
        
        self.is_fitted = True
        # Transform dopo il fit
        return self.pca.transform(data_array)

    def extract_and_save_features(self, data_loader, save_path):
        """Estrae le feature PCA da tutto il loader e le salva in .npz."""
        print(f"Estrazione feature PCA in corso -> {save_path}")
        all_features, all_labels = [], []
        for images, labels in data_loader:
            all_features.append(self.transform(images))
            all_labels.append(labels.numpy())
        features_np = np.vstack(all_features)
        labels_np   = np.vstack(all_labels)
        np.savez(save_path, features=features_np, labels=labels_np)
        print(f"Salvato: {features_np.shape}")

    def save_features(self, save_path, features, labels):
        np.savez(save_path, features=features, labels=labels)
        print(f"Feature salvate in {save_path}")

    def load_features(self, path):
        data = np.load(path)
        return {'features': data['features'], 'labels': data['labels']}

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.pca, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            self.pca = pickle.load(f)
        self.is_fitted = True

    @property
    def explained_variance_ratio_(self):
        return self.pca.explained_variance_ratio_


class ResNetCompressor(nn.Module):
    """
    Backbone ResNet18 con pesi pre-addestrati su MedMNIST.

    Parametri
    ---------
    data_flag      : nome del dataset MedMNIST (es. 'octmnist')
    as_rgb         : True se il DataLoader produce immagini a 3 canali.
                     Deve corrispondere esattamente all'impostazione usata in
                     data_loader.py. Default: True.
    weights_folder : cartella locale dove salvare/cercare i pesi.

    — mismatch canali
    ----------------------------
    INFO['octmnist']['n_channels'] == 1 (risoluzione nativa del dataset).
    Con as_rgb=True nel DataLoader le immagini arrivano però a 3 canali.
    Il vecchio codice confrontava n_channels con i canali del checkpoint
    e modificava conv1 per accettare 1 canale, crashando poi al forward
    (o producendo risultati silenziosamente sbagliati) perché le immagini
    reali erano a 3 canali.

    Soluzione: `as_rgb` è la fonte di verità sui canali effettivi in ingresso.
    Il confronto con il checkpoint avviene su `input_channels`, non su
    INFO['n_channels'].
    """

    BACKBONE_DIM = 512

    def __init__(self, data_flag, as_rgb=True, weights_folder="./weights"):
        super().__init__()

        info      = INFO[data_flag]
        n_classes = len(info['label'])

        # BUG 1 FIX: canali reali = 3 se as_rgb, altrimenti canali nativi del dataset
        input_channels = 3 if as_rgb else info['n_channels']

        resnet    = models.resnet18(weights=None)
        resnet.fc = nn.Linear(resnet.fc.in_features, n_classes)

        weights_path = self._prepare_weights(data_flag, weights_folder)
        if weights_path:
            state_dict = torch.load(weights_path, map_location='cpu')
            if 'net' in state_dict:
                state_dict = state_dict['net']

            ckpt_conv1 = state_dict.get('conv1.weight')
            if ckpt_conv1 is not None and ckpt_conv1.shape[1] != input_channels:
                ckpt_ch = ckpt_conv1.shape[1]
                if ckpt_ch == 3 and input_channels == 1:
                    print("Adattamento conv1: 3 canali → 1 canale (media dei pesi)")
                    state_dict['conv1.weight'] = ckpt_conv1.mean(dim=1, keepdim=True)
                    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                elif ckpt_ch == 1 and input_channels == 3:
                    print("Adattamento conv1: 1 canale → 3 canali (replica dei pesi)")
                    state_dict['conv1.weight'] = ckpt_conv1.repeat(1, 3, 1, 1) / 3.0
                    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                else:
                    raise ValueError(
                        f"Mismatch canali non gestito: checkpoint={ckpt_ch}, atteso={input_channels}"
                    )

            resnet.load_state_dict(state_dict)
            print("Pesi caricati con successo!")
        else:
            print("ATTENZIONE: pesi non trovati, il backbone usa pesi casuali.")

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.backbone.parameters():
            param.requires_grad = False

        # La proiezione è opzionale; non viene usata nell'estrazione delle feature.
        self.projection = None

    # ------------------------------------------------------------------
    # Estrazione feature (backbone puro, 512-dim)
    # ------------------------------------------------------------------

    def extract_backbone(self, x):
        """Restituisce il vettore 512-dim dal backbone (senza proiezione)."""
        with torch.no_grad():
            feat = self.backbone(x)
            feat = torch.flatten(feat, 1)
        return feat

    def forward(self, x):
        feat = self.extract_backbone(x)
        if self.projection is not None:
            feat = self.projection(feat)
        return feat

    def extract_and_save_features(self, data_loader, save_path, device="cpu"):
        """
        Salva le feature a 512 dimensioni (backbone puro, senza proiezione).

        BUG 3 FIX: np.vstack viene chiamato una sola volta per array;
        il risultato viene riusato per il salvataggio e per la stampa,
        eliminando la riallocazione inutile dell'intero array.
        """
        self.eval()
        self.to(device)
        all_features, all_labels = [], []

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                all_features.append(self.extract_backbone(images).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # BUG 3 FIX: vstack una volta sola, risultato riusato
        features_np = np.vstack(all_features)
        labels_np   = np.vstack(all_labels)

        np.savez(save_path, features=features_np, labels=labels_np)
        print(f"Feature backbone salvate: {save_path}  shape={features_np.shape}")

    def load_features(self, path):
        data = np.load(path)
        return {'features': data['features'], 'labels': data['labels']}

    # ------------------------------------------------------------------
    # Download e preparazione pesi MedMNIST
    # ------------------------------------------------------------------

    def _prepare_weights(self, data_flag, folder):
        os.makedirs(folder, exist_ok=True)
        extracted_pth = os.path.join(folder, f"{data_flag}_resnet18_224.pth")

        if os.path.exists(extracted_pth):
            return extracted_pth

        zip_filename = f"weights_{data_flag}.zip"
        zip_url      = f"https://zenodo.org/records/7782114/files/{zip_filename}?download=1"
        zip_path     = os.path.join(folder, zip_filename)

        try:
            if not os.path.exists(zip_path):
                print(f"Scaricamento pesi per {data_flag}...")
                response = requests.get(zip_url, stream=True)
                response.raise_for_status()
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)

            internal_file = None
            print(f"Estrazione da {zip_filename}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for name in zf.namelist():
                    if 'resnet18' in name and name.endswith('.pth'):
                        internal_file = name
                        zf.extract(name, folder)
                        break

            if internal_file:
                source = os.path.join(folder, internal_file)
                if source != extracted_pth:
                    if os.path.exists(extracted_pth):
                        os.remove(extracted_pth)
                    os.rename(source, extracted_pth)
                try:
                    os.remove(zip_path)
                except OSError:
                    pass
                return extracted_pth
            else:
                print("File .pth non trovato nell'archivio.")
                return None

        except Exception as e:
            print(f"Errore durante il download/estrazione: {e}")
            return None