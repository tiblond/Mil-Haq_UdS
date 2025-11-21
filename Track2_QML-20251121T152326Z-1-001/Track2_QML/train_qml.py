# train_qml.py
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Vos imports existants
from dataloaderLSTM import get_lstm_ready_data
from quantum_layer import QuantumPipeline # Le fichier créé ci-dessus

def main():
    # 1. Charger les données (On réutilise votre loader existant !)
    print("--- Chargement des données ---")
    X, y, scaler_y = get_lstm_ready_data('train.xlsx', seq_length=10)
    
    # Split (80/20)
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # 2. Pipeline Quantique
    print("\n--- Initialisation du Pipeline QML ---")
    # On utilise 4 modes (qubits/photons) pour commencer léger
    q_pipe = QuantumPipeline(n_modes=4, n_layers=2)
    
    print("Fit PCA...")
    q_pipe.fit_pca(X_train) # Apprendre la compression sur le train

    print("Transformation Quantique du Train set...")
    X_train_q = q_pipe.transform(X_train)
    
    print("Transformation Quantique du Val set...")
    X_val_q = q_pipe.transform(X_val)

    # 3. Apprentissage Classique (Readout)
    print("\n--- Entraînement du Readout (Ridge) ---")
    # Le réservoir est fixe, on apprend juste à lire la sortie avec une régression
    readout = Ridge(alpha=1.0)
    readout.fit(X_train_q, y_train)
    
    # 4. Évaluation
    print("\n--- Évaluation ---")
    preds_q = readout.predict(X_val_q)
    
    # Dénormalisation pour avoir des vrais dollars
    preds_dollar = scaler_y.inverse_transform(preds_q)
    y_true_dollar = scaler_y.inverse_transform(y_val)
    
    mae = np.mean(np.abs(preds_dollar - y_true_dollar))
    print(f"QML MAE : {mae:.4f} $")

    # Plot simple
    plt.plot(y_true_dollar[:, 0], label='Réel (Point 0)')
    plt.plot(preds_dollar[:, 0], label='Prédiction QML')
    plt.legend()
    plt.title("Résultat QML vs Réalité")
    plt.show()

if __name__ == "__main__":
    main()