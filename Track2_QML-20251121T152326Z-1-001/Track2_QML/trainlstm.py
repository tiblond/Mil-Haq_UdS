import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- VOS IMPORTS ---
from dataloaderLSTM import get_lstm_ready_data 
from model import smallLSTM                    

# --- PARAMÈTRES ---
SEQ_LENGTH = 10     
HIDDEN_DIM = 128
NUM_LAYERS = 2
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.001
DAYS_TO_PREDICT = 10 

def predict_future(model, last_sequence_scaled, days_to_predict=10):
    """
    Génère des prédictions jour par jour en réutilisant ses propres sorties.
    """
    future_predictions = []
    current_seq = last_sequence_scaled.clone() 
    
    model.eval()
    with torch.no_grad():
        for _ in range(days_to_predict):
            next_day_pred = model(current_seq) 
            future_predictions.append(next_day_pred.cpu().numpy())
            next_input = next_day_pred.unsqueeze(1) 
            current_seq = torch.cat((current_seq[:, 1:, :], next_input), dim=1)
            
    return np.array(future_predictions).squeeze()


def main():
    # ==========================================
    # 1. CHARGEMENT ET SPLIT
    # ==========================================
    print(f">>> 1. Préparation des données enrichies...")
    # On charge les données (X contient les indicateurs techniques, y les cibles)
    X, y, scaler_y = get_lstm_ready_data('train.xlsx', seq_length=SEQ_LENGTH)
    
    # Split Chronologique (Train 80% / Val 20%)
    split_idx = int(len(X) * 0.8)
    
    X_train = torch.FloatTensor(X[:split_idx])
    y_train = torch.FloatTensor(y[:split_idx])
    X_val = torch.FloatTensor(X[split_idx:])
    y_val = torch.FloatTensor(y[split_idx:])
    
    print(f"   Train set: {X_train.shape} samples")
    print(f"   Val set:   {X_val.shape} samples")
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    # ==========================================
    # 2. CONFIGURATION DU MODÈLE
    # ==========================================
    model = smallLSTM(
        input_shape=X.shape[2], 
        hidden=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_dim=y.shape[1]
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # ==========================================
    # 3. ENTRAÎNEMENT
    # ==========================================
    print(f">>> 2. Démarrage de l'entraînement ({EPOCHS} epochs)...")
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        # --- Train ---
        model.train() 
        batch_train_losses = []
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())
        
        avg_train_loss = np.mean(batch_train_losses)
        train_losses.append(avg_train_loss)
        
        # --- Validation (Loss uniquement) ---
        model.eval() 
        batch_val_losses = []
        
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                val_out = model(batch_X_val)
                v_loss = criterion(val_out, batch_y_val)
                batch_val_losses.append(v_loss.item())
        
        avg_val_loss = np.mean(batch_val_losses)
        val_losses.append(avg_val_loss)
        
        if (epoch+1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{EPOCHS} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")

    # Sauvegarde du modèle
    torch.save(model.state_dict(), 'lstm_final_track2.pth')

    # ==========================================
    # 4. VALIDATION FINALE : VRAIS PRIX (Ce que vous avez demandé)
    # ==========================================
    print(f"\n>>> 3. Vérification des prédictions (Prix Réels)...")
    
    model.eval()
    with torch.no_grad():
        val_preds_scaled = model(X_val).numpy()
        
    # Dénormalisation (On coupe à 224 colonnes au cas où le modèle en sort plus)
    # Cela permet de comparer des pommes avec des pommes (Vrais Dollars/Volatilité)
    val_preds_real = scaler_y.inverse_transform(val_preds_scaled[:, :224])
    y_val_real = scaler_y.inverse_transform(y_val.numpy()[:, :224])
    
    print(f"   {'Index':<6} | {'Prédiction':<12} | {'Réalité':<12} | {'Écart':<12}")
    print("-" * 55)
    # On affiche les 5 premiers jours du set de validation
    for k in range(5):
        pred_p = val_preds_real[k, 0] # On regarde la 1ère feature (souvent la plus importante)
        true_p = y_val_real[k, 0]     
        diff = pred_p - true_p
        print(f"   {k:<6} | {pred_p:.5f}      | {true_p:.5f}      | {diff:.5f}")

    # ==========================================
    # 5. PRÉPARATION DE LA SOUMISSION (XLSX)
    # ==========================================
    print(f"\n>>> 4. Génération du fichier de soumission...")
    
    template_path = 'template_track2_results.xlsx'
    submission_df = None
    days_to_predict = DAYS_TO_PREDICT 
    
    # Tentative de chargement du template
    try:
        submission_df = pd.read_excel(template_path)
        days_to_predict = len(submission_df)
        print(f"   Template chargé. Génération pour {days_to_predict} jours.")
    except FileNotFoundError:
        print(f"   Note : '{template_path}' introuvable. Mode défaut ({days_to_predict} jours).")

    # Prédiction auto-régressive du futur
    last_known_sequence = torch.FloatTensor(X[-1]).unsqueeze(0) 
    future_scaled = predict_future(model, last_known_sequence, days_to_predict=days_to_predict)
    
    if future_scaled.ndim == 1: 
        future_scaled = future_scaled[np.newaxis, :]

    # Dénormalisation pour la soumission
    future_prices = scaler_y.inverse_transform(future_scaled[:, :224])
    
    # Sauvegarde
    if submission_df is not None:
        if submission_df.shape[1] - 1 == future_prices.shape[1]:
            submission_df.iloc[:, 1:] = future_prices
            submission_df.to_excel('submission_results.xlsx', index=False)
            print(f"   >>> SUCCÈS ! 'submission_results.xlsx' créé.")
        else:
            print(f"   ERREUR DIMENSIONS : Template={submission_df.shape[1]-1} cols, Pred={future_prices.shape[1]} cols.")
            pd.DataFrame(future_prices).to_csv('debug_pred.csv', index=False)
    else:
        pd.DataFrame(future_prices).to_csv('future_predictions.csv', index=False)
        print("   Fichier 'future_predictions.csv' généré.")

    # Plot Final
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title("Training Log")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()