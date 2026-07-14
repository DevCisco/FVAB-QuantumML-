import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes


class QuantumPipeline:
    """
    Circuito VQC con DATA RE-UPLOADING multi-ciclo, PESI ANSATZ INDIPENDENTI
    per ciclo.

    Conforme all'interfaccia di encoding congelata del documento iniziale
    (sezione 4, "Interfaccia compressione — encoding"):
        "Re-upload schedule: blocchi da n_qubit feature per ciclo"
        "Padding: zero-padding se d != 0 (mod n_qubit)"
        "...si dividono le feature in blocchi, si caricano progressivamente
        sugli STESSI QUBIT e si alternano blocchi di encoding e blocchi
        variazionali."

    STORIA DELLA DECISIONE:
    E' stata testata anche la variante a pesi CONDIVISI tra i cicli (stesso
    ParameterVector "theta" riusato ad ogni blocco -- lo schema del paper
    Perez-Salinas et al., citato dal documento come riferimento per
    l'encoding). Verificato con un test diagnostico dedicato che il
    conteggio dei parametri nel circuito coincideva esattamente con quello
    atteso (nessun bug di implementazione in qc.compose()). Empiricamente,
    però, i pesi condivisi hanno PEGGIORATO le prestazioni rispetto ai pesi
    indipendenti (macro-F1 test sceso a 0.08-0.31, sotto il livello
    casuale) -- costringere lo stesso piccolo set di parametri a funzionare
    bene su blocchi di feature statisticamente diversi (specialmente a
    d=32, dove lo stesso blocco deve adattarsi a 8 segmenti diversi del
    vettore compresso) si è rivelato un vincolo piu difficile da
    soddisfare per NFT, non più facile.

    SCELTA FINALE: pesi INDIPENDENTI per ciclo (un ParameterVector "theta"
    di lunghezza n_cycles * parametri_per_blocco). Il numero di parametri
    variazionali scala quindi con d_latent (d=32 -> piu cicli -> piu
    parametri) -- gestito lato ottimizzatore da get_max_evals_nft in
    train_vqc_production.py, che scala il budget NFT di conseguenza.
    """

    def __init__(self, n_qubits=4, d_latent=4, n_layers=1):
        self.n_qubits = n_qubits
        self.d_latent = d_latent
        self.n_layers = n_layers

        # numero di cicli di re-upload: ceil(d_latent / n_qubits)
        self.n_cycles = int(np.ceil(d_latent / n_qubits))

        # slot di encoding totali (con zero-padding se d_latent non e
        # multiplo di n_qubits) -- lunghezza del ParameterVector "u".
        # Con D={32,16,8,4} e n_qubits=4 la divisione e sempre esatta
        # (n_padding=0), ma la logica resta generica e corretta per
        # qualunque combinazione (d_latent, n_qubits).
        self.n_encoding_padded = self.n_cycles * n_qubits
        self.n_padding = self.n_encoding_padded - d_latent

        # parametri di encoding: uno per ogni slot (reale o di padding)
        self.features = ParameterVector("u", self.n_encoding_padded)

        # template ansatz -- la STRUTTURA è identica per ogni ciclo, ma
        # ogni ciclo ha i SUOI parametri indipendenti (vedi build_circuit).
        self._ansatz_template = RealAmplitudes(
            num_qubits=n_qubits,
            reps=n_layers,
            entanglement='linear'
        )
        self._params_per_block = self._ansatz_template.num_parameters
        self.weights = ParameterVector(
            "theta", self.n_cycles * self._params_per_block
        )

        print(
            f"Circuito re-upload: {n_qubits} qubit | {self.n_cycles} cicli | "
            f"encoding={d_latent} feature (+{self.n_padding} padding) | "
            f"{len(self.weights)} pesi variazionali INDIPENDENTI "
            f"({self._params_per_block}/ciclo)"
        )

    def build_circuit(self):
        """
        Costruisce il circuito con n_cycles blocchi encoding+ansatz alternati:

            [RY block 1] [ansatz block 1] [RY block 2] [ansatz block 2] ...

        Ogni ciclo: RY su n_qubits feature (blocco corrente) -> ansatz
        RealAmplitudes con parametri PROPRI del ciclo (non condivisi).

        Decomposto in gate primitivi per compatibilita con Aer statevector.
        """
        qc = QuantumCircuit(self.n_qubits)
        params_per_block = self._ansatz_template.num_parameters

        for cycle in range(self.n_cycles):
            # blocco di encoding: n_qubits feature (reali o di padding)
            offset = cycle * self.n_qubits
            for i in range(self.n_qubits):
                qc.ry(self.features[offset + i], i)

            # blocco ansatz indipendente per questo ciclo
            w_offset = cycle * params_per_block
            ansatz_param_map = {
                p: self.weights[w_offset + i]
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
        qp = QuantumPipeline(n_qubits=4, d_latent=d, n_layers=2)
        circuit = qp.build_circuit()
        print(f"d={d}: {circuit.num_qubits} qubit, "
              f"{circuit.num_parameters} parametri, "
              f"profondita {circuit.depth()}\n")