import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes


class QuantumPipeline:
    """
    Circuito VQC a 4 qubit con un solo layer di encoding + un solo ansatz.

    Indipendentemente da d_latent, il circuito ha SEMPRE 4 qubit e 1 ansatz.
    Se d_latent > n_qubits, le feature vengono compresse con zero-padding
    (le prime n_qubits vengono codificate, le restanti ignorate a livello
    circuitale — la riduzione dimensionale è già avvenuta nella PCA).

    Questo garantisce che il simulatore statevector lavori sempre su
    2^4 = 16 ampiezze, rendendo la simulazione fattibile su 8 GB di RAM.

    Nota: il bando specifica n_qubits=4 e RealAmplitudes shallow come baseline.
    Il data re-uploading multi-ciclo è una variante esplorativa opzionale;
    per il benchmark comune si usa il circuito minimo qui definito.
    """

    def __init__(self, n_qubits=4, d_latent=4, n_layers=1):
        self.n_qubits = n_qubits
        self.d_latent = d_latent
        self.n_layers = n_layers

        # encoding: usiamo min(d_latent, n_qubits) feature per gate RY
        # le feature extra vengono ignorate a livello circuitale
        self.n_encoding = min(d_latent, n_qubits)

        # parametri di encoding (solo quelli effettivamente usati nel circuito)
        self.features = ParameterVector("u", self.n_encoding)

        # ansatz RealAmplitudes — un solo layer, sempre 4 qubit
        self._ansatz_template = RealAmplitudes(
            num_qubits=n_qubits,
            reps=n_layers,
            entanglement='linear'
        )
        self.weights = ParameterVector("theta", self._ansatz_template.num_parameters)

        print(f"Circuito: {n_qubits} qubit | encoding={self.n_encoding} feature "
              f"| {self._ansatz_template.num_parameters} pesi variazionali")

    def build_circuit(self):
        """
        Costruisce il circuito con un solo layer encoding + un solo ansatz.
        Decomposto in gate primitivi per compatibilità con Aer statevector.
        """
        qc = QuantumCircuit(self.n_qubits)

        # encoding RY su n_encoding qubit (≤ n_qubits)
        for i in range(self.n_encoding):
            qc.ry(self.features[i], i)

        # ansatz variazionale con mappatura esplicita dei parametri
        ansatz_param_map = {
            p: self.weights[i]
            for i, p in enumerate(
                sorted(self._ansatz_template.parameters, key=lambda p: p.name)
            )
        }
        qc.compose(
            self._ansatz_template.assign_parameters(ansatz_param_map),
            inplace=True
        )

        # decomposizione in gate primitivi riconosciuti da Aer
        qc = qc.decompose().decompose()
        return qc


def min_max_scaler(u, new_min=0, new_max=2 * np.pi):
    u_min, u_max = u.min(), u.max()
    if u_max == u_min:
        return np.zeros_like(u)
    return (u - u_min) / (u_max - u_min) * (new_max - new_min) + new_min


if __name__ == "__main__":
    for d in [32, 16, 8, 4]:
        qp = QuantumPipeline(n_qubits=4, d_latent=d, n_layers=1)
        circuit = qp.build_circuit()
        print(f"d={d}: {circuit.num_qubits} qubit, "
              f"{circuit.num_parameters} parametri, "
              f"profondità {circuit.depth()}\n")