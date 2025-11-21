import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
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

# ==========================================
# 0. CUSTOM LOSS & DIFFERENTIABLE MODEL
# ==========================================

class DifferentiableBlackModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f, k, sigma, t, r):
        """
        Vectorized Black-Scholes.
        Supports broadcasting if f/sigma are (Batch, 1) and k/t are (Batch, 224).
        """
        # Ensure tensors
        if not isinstance(f, torch.Tensor): f = torch.tensor(f)
        if not isinstance(sigma, torch.Tensor): sigma = torch.tensor(sigma)
        
        # Clamp for numerical stability
        sigma = torch.clamp(sigma, min=1e-4)
        f = torch.clamp(f, min=1e-4)
        t = torch.clamp(torch.tensor(t, device=f.device), min=1e-4)
        
        # Calculate d1 and d2
        # Note: We add dimensions to sigma/f to allow broadcasting against K and T
        d1 = (torch.log(f / k) + ((sigma**2 / 2) * t)) / (sigma * torch.sqrt(t))
        d2 = (torch.log(f / k) - ((sigma**2 / 2) * t)) / (sigma * torch.sqrt(t))
        
        norm = Normal(0, 1)
        
        # Calculate Price (Call option formula)
        p = torch.exp(-r * t) * (f * norm.cdf(d1) - k * norm.cdf(d2))
        return p

class PhysicsInformedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bs_model = DifferentiableBlackModel()

    def forward(self, pred_params, true_price_real, static_params):
        """
        pred_params: Tensor [Batch, 2] 
                     Col 0: Forward Price Ratio (F / K)
                     Col 1: Volatility (Sigma)
        true_price_real: Tensor [Batch, N] (Actual Market Prices in $)
        """
        
        # 1. INTERPRET MODEL OUTPUTS
        # We assume the model predicts F relative to the Strike (e.g., 1.05 means 5% above strike)
        # This is more stable than predicting raw prices like 3500.0
        strike_ref = static_params['k']
        
        # F = (Predicted Ratio) * Strike
        pred_F = torch.abs(pred_params[:, 0:1]) * strike_ref 
        
        # Sigma = Predicted value (force positive)
        pred_sigma = torch.abs(pred_params[:, 1:2])
        
        # 2. GENERATE PRICES VIA PHYSICS LAYER
        # This will broadcast: (Batch, 1) params vs (Scalar or Vector) K/T
        pred_price = self.bs_model(
            f=pred_F,
            k=static_params['k'],
            sigma=pred_sigma,
            t=static_params['t'],
            r=static_params['r'],
        )
        
        # 3. COMPUTE LOSS
        # If true_price_real is (Batch, 224) and pred_price is (Batch, 224), this works.
        # If pred_price comes out as (Batch, 1), we might need to check dimensions.
        return self.mse(pred_price, true_price_real)


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print(f">>> 1. Préparation des données...")
    X, y, scaler_y = get_lstm_ready_data('train.xlsx', seq_length=SEQ_LENGTH)
    
    split_idx = int(len(X) * 0.8)
    X_train, y_train = torch.FloatTensor(X[:split_idx]), torch.FloatTensor(y[:split_idx])
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. CONFIGURATION
    # CRITICAL FIX: output_dim must be 2.
    # The error (16, 243) happened because you likely left this as X.shape[2] or similar.
    model = smallLSTM(
        input_shape=X.shape[2], 
        hidden=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_dim=2  # <--- FORCE THIS TO 2 (Predicting F and Sigma)
    )
    
    criterion = PhysicsInformedLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # STATIC PARAMS
    # Ideally, these should be vectors of size 224 to match your surface targets.
    # For now, we use scalars, which means the model tries to fit 1 price.
    # If 'y' is a surface, these MUST be vectors to work correctly.
    static_params = {
        'k': 100.0,   
        't': 1.0,      
        'r': 0.05      
    }

    print(f">>> 2. Démarrage de l'entraînement ({EPOCHS} epochs)...")
    
    for epoch in range(EPOCHS):
        model.train() 
        batch_losses = []
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # 1. Forward Pass -> Shape (Batch, 2)
            pred_params = model(batch_X)
            
            # 2. Inverse Transform Target -> Shape (Batch, 224) or (Batch, 1)
            # We need Real Prices to compare against Black-Scholes output
            batch_y_real = torch.tensor(scaler_y.inverse_transform(batch_y.numpy())).float()
            
            # 3. Calculate Loss
            # Note: We removed scaler_min/scale arguments. 
            loss = criterion(
                pred_params=pred_params,
                true_price_real=batch_y_real, 
                static_params=static_params
            )
            
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        avg_loss = np.mean(batch_losses)
        
        if (epoch+1) % 5 == 0:
            print(f"   Epoch {epoch+1} | Physics Loss (Real $ Error): {avg_loss:.6f}")

    print("   Training Complete.")

if __name__ == "__main__":
    main()