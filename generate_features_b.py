import numpy as np
import torch
from sklearn.decomposition import PCA

def extract_b_features(d, cache_512_train, cache_512_test, model_b2, model_b3):
    """
    Estrae le feature per i tre casi B1, B2, B3 da train e test set separati.

    FIX BUG 1: la versione originale accettava un unico 'cache_512' e veniva
    chiamata separatamente per train e test, rifittando la PCA ogni volta.
    Fittare la PCA sui dati di test è data leakage: la PCA apprende la
    distribuzione del test set e le metriche risultanti sono invalide.
    Ora la PCA viene fittata SOLO su cache_512_train e applicata (transform)
    su entrambi i set, come richiesto dalla correttezza sperimentale.

    FIX BUG 2: il controllo isinstance(b1_feat, torch.Tensor) era inutile e
    invertito — PCA.fit_transform restituisce sempre np.ndarray, mai Tensor.
    Rimosso il branch ridondante.
    """
    # B1: PCA — fit solo su train, transform su entrambi
    pca = PCA(n_components=d)
    b1_train = pca.fit_transform(cache_512_train)   # fit+transform sul train
    b1_test  = pca.transform(cache_512_test)         # solo transform sul test

    # B2 & B3: Autoencoder — il modello è già stato addestrato su train
    with torch.no_grad():
        _, b2_train = model_b2(torch.tensor(cache_512_train).float())
        _, b3_train = model_b3(torch.tensor(cache_512_train).float())
        _, b2_test  = model_b2(torch.tensor(cache_512_test).float())
        _, b3_test  = model_b3(torch.tensor(cache_512_test).float())

    return (
        {'B1': b1_train, 'B2': b2_train.numpy(), 'B3': b3_train.numpy()},
        {'B1': b1_test,  'B2': b2_test.numpy(),  'B3': b3_test.numpy()}
    )