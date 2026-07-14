import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from qiskit_aer.primitives import EstimatorV2
from qiskit.quantum_info import SparsePauliOp

from quantum_model import QuantumPipeline
from pca_res_compressors import PCACompressor, ResNetCompressor

AS_RGB   = True
FLAG     = 'octmnist'
N_FIT    = 10_000
N_QUBITS = 4


class DirectVQC(nn.Module):
    """
    Layer quantistico che chiama EstimatorV2 direttamente, senza
    EstimatorQNN né TorchConnector.

    I pesi variazionali sono un nn.Parameter standard: SPSA li legge e
    li imposta tramite get_flat_params/set_flat_params come qualsiasi
    altro parametro PyTorch.

    Il forward è una funzione numpy pura wrappata in
    torch.autograd.Function per compatibilità con il grafo PyTorch
    quando serve (es. loss.backward() fuori da SPSA).
    Con SPSA + torch.no_grad() è una semplice chiamata numpy → tensor.
    """

    def __init__(self, circuit, features_pv, weights_pv, n_qubits, estimator):
        super().__init__()
        self.circuit     = circuit
        self.features_pv = features_pv      # ParameterVector input
        self.n_qubits    = n_qubits
        self.estimator   = estimator

        # indici nel vettore parametri globale del circuito
        all_params = list(features_pv) + list(weights_pv)
        self._feat_len    = len(features_pv)
        self._weight_len  = len(weights_pv)

        # osservabili: Pauli-Z su ogni qubit
        self.obs = [
            SparsePauliOp("I" * i + "Z" + "I" * (n_qubits - i - 1))
            for i in range(n_qubits)
        ]

        # pesi variazionali come parametro PyTorch ottimizzabile
        # Init U(-0.01, 0.01) — permette di partire vicino a 0 e quindi vicino a identità, utile per l'ottimizzazione.
        self.qweights = nn.Parameter(
            (torch.rand(self._weight_len, dtype=torch.float32) * 2 - 1) * 0.01
        )

    def _run_estimator(self, feat_batch: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Esegue EstimatorV2 su un batch di feature.
        Restituisce array (batch_size, n_qubits) di expectation values.
        """
        pub_list = [
            (self.circuit, self.obs, np.concatenate([feat_batch[i], weights]))
            for i in range(len(feat_batch))
        ]
        result = self.estimator.run(pub_list).result()
        return np.array([
            [result[i].data.evs[j] for j in range(self.n_qubits)]
            for i in range(len(feat_batch))
        ], dtype=np.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_np    = x.detach().cpu().numpy()
        weight_np  = self.qweights.detach().cpu().numpy()
        evs        = self._run_estimator(feat_np, weight_np)
        out        = torch.tensor(evs, dtype=torch.float32, device=x.device)
        # ricollega al grafo PyTorch per compatibilità con loss.backward()
        # (con SPSA e torch.no_grad() questo è un no-op)
        return out + 0.0 * self.qweights.sum()


class HybridModel(nn.Module):
    """
    Pipeline: immagine → ResNet-512 → StandardScaler → PCA d-dim
              → scaling [0,π] → DirectVQC → Linear → 4 classi.

    Parametri ottimizzati da SPSA: solo qweights (8) + classifier (20) = 28.
    Il backbone ResNet (~11.2M) ha requires_grad=False ed è ESCLUSO
    dall'ottimizzazione — questo era la causa del crash da 909 TiB.

    Config keys:
        d_latent (int): 32, 16, 8, 4
        n_qubits (int): 4
        n_layers (int): 1
        seed     (int): seed usato in test.py
    """

    RAW_PATH = "artifacts/resnet/features/res_raw_d_{d}_{split}_s{seed}.npz"

    def __init__(self, config: dict, backbone_type: str = 'resnet'):
        super().__init__()
        self.d_latent = config['d_latent']
        self.n_qubits = config.get('n_qubits', N_QUBITS)
        seed          = config['seed']

        # ── 1. Backbone congelato (requires_grad=False su tutti i param) ─
        self.backbone = ResNetCompressor(data_flag=FLAG, as_rgb=AS_RGB)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ── 2. Scaler + PCA su sottoinsieme train ─────────────────────
        train_path = self.RAW_PATH.format(d=self.d_latent, split='train', seed=seed)
        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"File raw non trovato: {train_path}\n"
                "Esegui prima test.py per generarlo."
            )

        x_all = np.load(train_path)['features']
        idx   = np.random.default_rng(42).choice(
            len(x_all), size=min(N_FIT, len(x_all)), replace=False
        )
        x_fit = x_all[idx];  del x_all

        self.scaler = StandardScaler()
        x_fit_scaled = self.scaler.fit_transform(x_fit)

        self.pca = PCACompressor(n_components=self.d_latent)
        self.pca.fit_transform(x_fit_scaled)
        del x_fit, x_fit_scaled

        # ── 3. Circuito + DirectVQC ───────────────────────────────────
        self.q_pipeline = QuantumPipeline(
            n_qubits=self.n_qubits,
            d_latent=self.d_latent,
            n_layers=config.get('n_layers', 1),
        )
        circuit = self.q_pipeline.build_circuit()

        self.vqc = DirectVQC(
            circuit     = circuit,
            features_pv = self.q_pipeline.features,
            weights_pv  = self.q_pipeline.weights,
            n_qubits    = self.n_qubits,
            estimator   = EstimatorV2(),
        )

        # ── 4. Testa lineare ──────────────────────────────────────────
        self.classifier = nn.Linear(self.n_qubits, 4)

    def _compress(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats_512 = self.backbone.extract_backbone(x).cpu().numpy()
        feats_scaled = self.scaler.transform(feats_512)
        feats_d      = self.pca.pca.transform(feats_scaled)
        n_enc        = self.q_pipeline.n_encoding
        return torch.tensor(feats_d[:, :n_enc], dtype=torch.float32).to(x.device)

    def scale_features(self, u: torch.Tensor) -> torch.Tensor:
        u_min = u.min(dim=1, keepdim=True)[0]
        u_max = u.max(dim=1, keepdim=True)[0]
        return (u - u_min) / (u_max - u_min + 1e-8) * np.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u        = self._compress(x)
        u_scaled = self.scale_features(u)
        q_out    = self.vqc(u_scaled)
        return self.classifier(q_out)


if __name__ == "__main__":
    config = {'d_latent': 8, 'n_qubits': 4, 'n_layers': 1, 'seed': 11}
    model  = HybridModel(config, backbone_type='resnet')
    # verifica che i parametri ottimizzabili siano solo 28
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri ottimizzabili: {trainable}")  # atteso: 28
    out = model(torch.randn(2, 3, 224, 224))
    print(f"Output shape: {out.shape}")