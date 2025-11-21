# train_hybrid_black.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler

# Vos imports locaux
from dataloaderLSTM import get_lstm_ready_data
from quantum_layer import QuantumPipeline
from black import Loss as BlackLoss # Votre fichier black.py

# --- CONFIGURATION ---
LR = 0.01
EPOCHS = 20 # Moins d'époques car on manque de temps
N_MODES = 4 # Garder petit pour aller vite

# --- PETIT RÉSEAU POUR REMPLACER LA RIDGE ---
# Il prend les features quantiques et prédit les paramètres pour Black-Scholes
class ReadoutNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Couche linéaire simple (équivalent Ridge) mais entraînable par gradient
        self.linear = nn.Linear(input_dim, 2) # Sortie: 2 params (F, Sigma)
        
    def forward(self, x):
        # On force des valeurs positives pour Sigma (Volatilité)
        return torch.abs(self.linear(x)) 

def main():
    # 1. Charger les données
    print(">>> 1. Chargement Data...")
    X, y, scaler_y = get_lstm_ready_data('train.xlsx', seq_length=10)
    
    # On prend un sous-ensemble pour aller vite (ex: les 100 derniers jours)
    # Si vous avez le temps, prenez tout.
    X_small = X[-100:]
    y_small = y[-100:]

    # 2. Générer les Features Quantiques (RESERVOIR FIXE)
    print(">>> 2. Génération Features Quantiques...")
    q_pipe = QuantumPipeline(n_modes=N_MODES)
    q_pipe.fit_pca(X_small)
    X_quantum = q_pipe.transform(X_small) # C'est là que le QPU travaille
    
    # Conversion en Tenseurs
    X_tensor = torch.FloatTensor(X_quantum)
    y_tensor = torch.FloatTensor(y_small) # Prix réels
    
    # 3. Initialisation du Readout et de la Loss Black
    # Attention: BlackLoss attend des "static_params" (K, T, r)
    # Pour le hackathon, on fixe des valeurs moyennes si on ne les a pas colonne par colonne
    static_params = {'k': 1.0, 't': 1.0, 'r': 0.05} # VALEURS PAR DÉFAUT À ADAPTER
    
    # On a besoin des stats pour la Loss Black (voir votre fichier black.py)
    scaler_mean = torch.tensor([0.0]) 
    scaler_std = torch.tensor([1.0])

    model = ReadoutNet(input_dim=X_quantum.shape[1], output_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = BlackLoss()

    # 4. Boucle d'Entraînement Hybride
    print(">>> 3. Entraînement avec Physics-Informed Loss...")
    model.train()
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # A. Le réseau prédit (F, Sigma) à partir des états quantiques
        pred_params = model(X_tensor) 
        
        # B. La Loss Black calcule le prix théorique et compare au vrai prix y_tensor
        # Note: Votre Loss Black semble être faite pour 1 output (Batch, 1).
        # Si y_tensor est (Batch, 224), il faudra adapter ou prendre la moyenne.
        
        # Pour le défi Track 2 (Surface), on va simplifier et calculer la MSE globale
        # en utilisant Black comme régularisateur ou sur une moyenne.
        # ICI: Hack pour faire marcher votre code black.py tel quel :
        # On suppose qu'on prédit un prix moyen représentatif
        y_mean_price = y_tensor.mean(dim=1, keepdim=True) 
        
        loss = loss_fn(pred_params, y_mean_price, static_params, scaler_mean, scaler_std)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Black Loss: {loss.item():.6f}")

    print(">>> TERMINÉ. Modèle Hybride Entraîné.")
    
    # Sauvegarde rapide
    torch.save(model.state_dict(), 'hybrid_black_model.pth')

if __name__ == "__main__":
    main()