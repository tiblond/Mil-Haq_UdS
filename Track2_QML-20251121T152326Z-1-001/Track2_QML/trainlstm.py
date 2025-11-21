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
    X, y, scaler_y = get_lstm_ready_data('train.xlsx', seq_length=SEQ_LENGTH) #
    
    # --- SPLIT CHRONOLOGIQUE ---
    split_idx = int(len(X) * 0.8)
    
    X_train = torch.FloatTensor(X[:split_idx])
    y_train = torch.FloatTensor(y[:split_idx])
    X_val = torch.FloatTensor(X[split_idx:])
    y_val = torch.FloatTensor(y[split_idx:])
    
    print(f"   Train set: {X_train.shape}")
    print(f"   Val set:   {X_val.shape}")
    
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
    # 3. ENTRAÎNEMENT AVEC VALIDATION
    # ==========================================
    print(f">>> 2. Démarrage de l'entraînement ({EPOCHS} epochs)...")
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        # --- A. Phase d'entraînement ---
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
        
        # --- B. Phase de validation avec PRINTS ---
        model.eval() 
        batch_val_losses = []
        
        # Variables pour stocker un exemple
        example_pred = None
        example_true = None
        
        with torch.no_grad():
            for i, (batch_X_val, batch_y_val) in enumerate(val_loader):
                val_out = model(batch_X_val)
                v_loss = criterion(val_out, batch_y_val)
                batch_val_losses.append(v_loss.item())
                
                # On capture le premier élément du premier batch pour affichage
                if i == 0:
                    example_pred = val_out[0, 0].item()      # 1er sample, 1ère feature
                    example_true = batch_y_val[0, 0].item()
        
        avg_val_loss = np.mean(batch_val_losses)
        val_losses.append(avg_val_loss)
        
        # Affichage propre tous les 5 epochs
        if (epoch+1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            # --- AJOUT: COMPARAISON VISUELLE (Valeurs normalisées) ---
            print(f"      -> Check (Norm): Pred={example_pred:.4f} vs True={example_true:.4f} | Diff={abs(example_pred-example_true):.4f}")

    # Sauvegarde
    torch.save(model.state_dict(), 'lstm_final_track2.pth')
    
    # ==========================================
    # 4. VISUALISATION FINALE (REAL PRICES)
    # ==========================================
    print(f"\n>>> 3. Validation Finale : Comparaison des Vrais Prix")
    
    model.eval()
    with torch.no_grad():
        val_preds_scaled = model(X_val).numpy()
        
    # --- CORRECTION HERE ---
    # The model outputs 243 columns (Prices + Technical Features).
    # scaler_y only knows the 224 Prices. We slice [:, :224].
    
    val_preds_real = scaler_y.inverse_transform(val_preds_scaled[:, :224])
    y_val_real = scaler_y.inverse_transform(y_val.numpy()[:, :224])
    
    print(f"   {'Index':<6} | {'Prédiction':<12} | {'Réalité':<12} | {'Écart':<12}")
    print("-" * 50)
    for k in range(5):
        pred_p = val_preds_real[k, 0] 
        true_p = y_val_real[k, 0]     
        diff = pred_p - true_p
        print(f"   {k:<6} | {pred_p:.5f}      | {true_p:.5f}      | {diff:.5f}")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.show()

    # ==========================================
    # 5. PRÉDICTION DU FUTUR
    # ==========================================
    print(f"\n>>> 4. Génération des prédictions futures ({DAYS_TO_PREDICT} jours)...")
    
    last_known_sequence = torch.FloatTensor(X[-1]).unsqueeze(0) 
    
    # The model will predict 243 features for the future too
    future_scaled = predict_future(model, last_known_sequence, days_to_predict=DAYS_TO_PREDICT)
    
    # --- CORRECTION HERE ---
    # Slice again to keep only the 224 price columns
    future_prices = scaler_y.inverse_transform(future_scaled[:, :224])
    
    print(f"   Exemple (J+1, Feature 0) : {future_prices[0][0]:.4f}")
    
    # Save only the 224 columns required
    pd.DataFrame(future_prices).to_csv('future_predictions.csv', index=False)
    print("   Fichier 'future_predictions.csv' généré.")

if __name__ == "__main__":
    main()