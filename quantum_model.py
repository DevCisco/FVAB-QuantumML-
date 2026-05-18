import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
# FIX #1: RealAmplitudes è una classe, non una funzione — nome corretto con maiuscole
from qiskit.circuit.library import real_amplitudes

class QuantumPipeline:
    def __init__(self, n_qubits, d_latent, n_layers=1):
        self.n_qubits = n_qubits
        self.d_latent = d_latent
        self.n_layers = n_layers
        
        self.re_upload_cycles = int(np.ceil(d_latent / n_qubits))
        print(self.re_upload_cycles, "cicli di re-uploading necessari per d_latent =", d_latent)
        
        self.features = ParameterVector("u", d_latent)
        
        # FIX #1 applicato: RealAmplitudes (classe) al posto di real_amplitudes (funzione inesistente)
        # Usiamo un ansatz "template" solo per ricavare num_parameters
        _ansatz_template = real_amplitudes(num_qubits=n_qubits, reps=n_layers, entanglement='linear')
        self.n_weights_per_cycle = _ansatz_template.num_parameters

        # FIX #6: ogni ciclo di re-uploading deve avere i PROPRI pesi distinti.
        # Con un unico ParameterVector condiviso tutti i layer variazionali usano
        # gli stessi parametri → gradi di libertà effettivi ridotti, gradienti errati.
        # Creiamo re_upload_cycles vettori separati "theta_c<k>" e un ansatz per ciclo.
        self.ansatz_list = []
        self.weights_list = []
        for k in range(self.re_upload_cycles):
            w = ParameterVector(f"theta_c{k}", self.n_weights_per_cycle)
            a = real_amplitudes(num_qubits=n_qubits, reps=n_layers, entanglement='linear')
            self.weights_list.append(w)
            self.ansatz_list.append(a)

        # Vettore piatto di tutti i pesi (usato da HybridModel per costruire weight_params)
        self.all_weights = [w[j] for w in self.weights_list for j in range(self.n_weights_per_cycle)]

    def build_circuit(self):
        qc = QuantumCircuit(self.n_qubits)
        
        feat_idx = 0
        for cycle in range(self.re_upload_cycles):
            # 1. Encoding Layer (RY Baseline)
            for i in range(self.n_qubits):
                if feat_idx < self.d_latent:
                    qc.ry(self.features[feat_idx], i)
                    feat_idx += 1
            
            # 2. Variational Layer — pesi distinti per questo ciclo (FIX #6)
            qc.compose(
                self.ansatz_list[cycle].assign_parameters(self.weights_list[cycle]),
                inplace=True
            )
            
            qc.barrier()
            
        return qc

def min_max_scaler(u, new_min=0, new_max=2*np.pi):
    """ Scala il vettore u nel range richiesto per i gate quantistici """
    u_min, u_max = u.min(), u.max()
    if u_max == u_min:
        return np.zeros_like(u)
    return (u - u_min) / (u_max - u_min) * (new_max - new_min) + new_min

if __name__ == "__main__":
    qp = QuantumPipeline(n_qubits=4, d_latent=32, n_layers=1)
    circuit = qp.build_circuit()
    print("Struttura del circuito (Data Re-uploading):")
    print(circuit.draw(output='text'))
    qp = QuantumPipeline(n_qubits=4, d_latent=16, n_layers=1)
    circuit = qp.build_circuit()
    print("Struttura del circuito (Data Re-uploading):")
    print(circuit.draw(output='text'))
    qp = QuantumPipeline(n_qubits=4, d_latent=8, n_layers=1)
    circuit = qp.build_circuit()
    print("Struttura del circuito (Data Re-uploading):")
    print(circuit.draw(output='text'))
    qp = QuantumPipeline(n_qubits=4, d_latent=4, n_layers=1)
    circuit = qp.build_circuit()
    print("Struttura del circuito (Data Re-uploading):")
    print(circuit.draw(output='text'))