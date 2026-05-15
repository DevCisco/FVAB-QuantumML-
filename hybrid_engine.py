import torch
import torch.nn as nn
import numpy as np
import pickle
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterVector

from quantum_model import QuantumPipeline
from pca_res_compressors import ResNetCompressor

flag = 'octmnist'

class HybridModel(nn.Module):
    def __init__(self, config, backbone_type='pca'):
        super(HybridModel, self).__init__()
        self.d_latent = config['d_latent']
        self.n_qubits = config['n_qubits']
        self.backbone_type = backbone_type

        # 1. Caricamento del Backbone (Congelato)
        self.backbone = ResNetCompressor(d_latent=self.d_latent, data_flag=flag)
        path = f"artifacts/resnet/resnet_propper_d{self.d_latent}.pth"
        # FIX #7: weights_only=True previene l'esecuzione di pickle arbitrario
        # (vulnerabilità RCE) ed è obbligatorio da PyTorch 2.4+.
        self.backbone.load_state_dict(torch.load(path, weights_only=True))
        self.backbone.eval() 
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Definizione del Circuito Quantistico
        self.q_pipeline = QuantumPipeline(
            n_qubits=self.n_qubits, 
            d_latent=self.d_latent,
            n_layers=config['n_layers']
        )
        self.circuit = self.q_pipeline.build_circuit()

        # 3. Osservabili — un Z per qubit, gli altri I (FIX #11)
        # In Qiskit le stringhe Pauli sono little-endian: il carattere più a DESTRA
        # corrisponde al qubit 0. La versione originale usava "Z" al posto di "I"
        # per i qubit non misurati → misure su qubit sbagliati, output scientificamente errato.
        # Corretto: "I"*i + "Z" + "I"*(n_qubits-i-1)
        obs = [SparsePauliOp("I" * i + "Z" + "I" * (self.n_qubits - i - 1))
               for i in range(self.n_qubits)]
        
        # FIX #2 aggiornato: i prefissi dei pesi sono ora "theta_c0[", "theta_c1[", ecc.
        # (grazie al FIX #6 in quantum_model.py che assegna un ParameterVector distinto
        # per ciclo). Selezioniamo tutti i parametri il cui nome NON inizia con "u[".
        all_params = self.circuit.parameters

        input_params_raw = sorted(
            [p for p in all_params if p.name.startswith("u[")],
            key=lambda p: int(p.name[2:-1])
        )
        weight_params = sorted(
            [p for p in all_params if not p.name.startswith("u[")],
            key=lambda p: p.name   # ordine lessicografico stabile per i theta_cK[j]
        )

        # Rinominiamo i parametri di input in 'x' come si aspetta EstimatorQNN
        input_params = ParameterVector('x', self.d_latent)
        param_map = {p: input_params[i] for i, p in enumerate(input_params_raw)}
        circuit_with_params = self.circuit.assign_parameters(param_map, inplace=False)
        
        # 4. EstimatorQNN
        self.qnn = EstimatorQNN(
            circuit=circuit_with_params,
            estimator=Estimator(),
            observables=obs,
            input_params=input_params,
            weight_params=weight_params
        )
        
        # 5. TorchConnector
        self.quantum_layer = TorchConnector(self.qnn)

        # FIX #8: self.pca_model non veniva mai inizializzato → AttributeError
        # garantito nel forward() quando backbone_type != 'resnet'.
        # Lo inizializziamo a None; chiamare set_pca_model() prima del forward.
        self.pca_model = None

    def set_pca_model(self, pca_model):
        """Assegna il modello PCA sklearn da usare come backbone alternativo."""
        self.pca_model = pca_model

    def scale_features(self, x):
        """ Scala l'input nel range [0, pi] per i gate RY """
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        return (x - x_min) / (x_max - x_min + 1e-8) * np.pi

    def forward(self, x):
        # A. Compressione
        if self.backbone_type == 'resnet':
            with torch.no_grad():
                u = self.backbone(x)
        else:
            # FIX #8: verifica esplicita che pca_model sia stato assegnato
            if self.pca_model is None:
                raise RuntimeError(
                    "backbone_type='pca' richiede di chiamare set_pca_model() "
                    "prima del forward pass."
                )
            x_flat = x.view(x.size(0), -1).cpu().numpy()
            u_np = self.pca_model.transform(x_flat)
            u = torch.tensor(u_np, dtype=torch.float32).to(x.device)

        # B. Scaling
        u_scaled = self.scale_features(u)

        # C. Quantum Forward Pass
        q_out = self.quantum_layer(u_scaled)

        return q_out

if __name__ == "__main__":
    config = {
        'd_latent': 8,
        'n_qubits': 4,
        'n_layers': 1
    }
    
    # FIX #8: il test usava backbone_type='pca' senza fornire self.pca_model →
    # crash garantito nel forward. Usiamo 'resnet' per il test di default,
    # che è il backbone effettivamente supportato senza setup aggiuntivo.
    model = HybridModel(config, backbone_type='resnet')
    dummy_input = torch.randn(2, 1, 224, 224)
    output = model(dummy_input)
    
    print(f"Output del modello ibrido (Expectation Values):\n{output}")
    print(f"Shape dell'output: {output.shape}")