# quantum_layer.py
import perceval as pcvl
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

# Ajoutez cet import
import os 




class QuantumPipeline:
    def __init__(self, n_modes=4, n_layers=2, max_shots_per_call=100):
        self.n_modes = n_modes
        # Nombre maximal de tirs par appel requis par certains RemoteProcessor
        self.max_shots_per_call = max_shots_per_call
        # PCA pour réduire les 224 features vers n_modes (limite du hardware quantique)
        self.pca = PCA(n_components=n_modes)
        
        # Modifiez la ligne self.processor = ... dans __init__
        # ---------------------------------------------------------
        # OPTION 1 : VRAI HARDWARE (Avoir un token valide)
        # "qpu:ascella" est un exemple de puce, vérifiez celle dispo sur votre compte
        try:
            # Remplacez VOTRE_TOKEN par votre vraie clé ou mettez-la en variable d'env
            token = os.getenv("QUANDELA_TOKEN", "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjAwMywiZXhwIjoxNzY2MzQ1MTU0LjY3ODcyNzZ9.3obyetHXpS-e816f5IEzv0k7CMb0jqQIlZqelMD2rPGQG0tyh9PhcP4I2jlQt53MR8zKH07qvxrHRYvu7G6ofQ") 
            self.processor = pcvl.RemoteProcessor("qpu:ascella", token)
            print(">>> CONNECTÉ AU QPU QUANDELA (REAL HARDWARE)")
        except Exception as e:
            print(f">>> ECHEC CONNEXION QPU ({e}). PASSAGE EN SIMULATION.")
            self.processor = pcvl.Processor("SLOS", n_modes)
        # ---------------------------------------------------------

        # Construction du "Réservoir" (Circuit aléatoire fixe)
        self.circuit = pcvl.Circuit(n_modes)
        for _ in range(n_layers):
            for i in range(n_modes - 1):
                # On mélange les modes avec des composants optiques
                self.circuit.add(i, pcvl.BS.H()) 
                self.circuit.add(i, pcvl.PS(phi=np.random.rand() * np.pi))
        
        self.processor.set_circuit(self.circuit)

    def fit_pca(self, X):
        """Calibre la réduction de dimension sur le train set"""
        # Aplatir les données si elles viennent du LSTM (Batch, Time, Feat) -> (Batch, Feat)
        if len(X.shape) == 3:
            X = X[:, -1, :] # On prend juste le dernier pas de temps
        self.pca.fit(X)

    def transform(self, X):
        """Transforme les données classiques en probabilités quantiques"""
        if len(X.shape) == 3:
            X = X[:, -1, :] # (Batch, Features)
            
        print(">>> 1. Encodage (PCA + Normalisation)...")
        X_reduced = self.pca.transform(X)
        # Normalisation entre 0 et Pi pour les angles des phases optiques
        X_norm = (X_reduced - X_reduced.min()) / (X_reduced.max() - X_reduced.min()) * np.pi
        
        print(">>> 2. Simulation Quantique (C'est long, patience !)...")
        quantum_features = []
        
        # État d'entrée : 1 photon dans chaque mode
        input_state = pcvl.BasicState([1] * self.n_modes)

        for sample in tqdm(X_norm):
            # Ici on pourrait encoder 'sample' dans le circuit (Data Re-uploading)
            # Pour ce MVP, on utilise le circuit comme un filtre complexe fixe

            # Certains processeurs distants exigent un paramètre `max_shots_per_call`.
            # On le fournit de façon explicite (compatible aussi avec les simulateurs).
            sampler = pcvl.algorithm.Sampler(self.processor, max_shots_per_call=self.max_shots_per_call)
            # On tire des échantillons pour avoir une distribution de proba
            probs = sampler.probs(input_state=input_state)['results']

            # On transforme le dictionnaire de résultats en vecteur
            # C'est une version simplifiée, on prend juste les valeurs brutes
            vec = list(probs.values())
            # Padding si nécessaire (pour avoir une taille fixe)
            target_len = 20 # Taille arbitraire de sortie
            if len(vec) < target_len:
                vec += [0] * (target_len - len(vec))

            quantum_features.append(vec[:target_len])

        return np.array(quantum_features)