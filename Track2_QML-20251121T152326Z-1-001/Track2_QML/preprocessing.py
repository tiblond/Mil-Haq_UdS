import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import des fonctions fournies par le hackathon
import data
def run_data_pipeline(file_path='train.xlsx', forecast_horizon=1):
    print(">>> 1. Chargement et Nettoyage initial")
    # data.py charge le fichier, trie par date et gère le décalage de cible (shift)
    # X contient les features du jour J, y contient les cibles du jour J+1
    X_raw, y_raw, dates = data.load_and_prepare_data(file_path, forecast_horizon)
    
    print(f"Données brutes chargées: {X_raw.shape}")

    # >>> 2. Feature Engineering (Crucial pour la finance)
    # Nous allons ajouter les valeurs passées (lags) comme features.
    # C'est essentiel pour que le modèle (Classique ou Quantique) comprenne la dynamique.
    print(">>> 2. Création des Lagged Features")
    
    # On utilise la fonction de data.py pour créer des lags de 1, 2, 3 et 5 jours
    # Cela permet au modèle de voir "ce qu'il s'est passé la semaine dernière"
    X_enhanced = data.create_lagged_features(X_raw, feature_cols=X_raw.columns, lags=[1, 2, 3, 5])
    
    # ATTENTION : Le lagging crée des NaN au début du fichier (on ne peut pas avoir de lag au jour 0).
    # Il faut aligner X et y en supprimant ces lignes vides.
    
    # On trouve les indices communs (ceux qui n'ont pas été supprimés par le lagging)
    common_index = X_enhanced.index.intersection(y_raw.index)
    
    X_final = X_enhanced.loc[common_index]
    y_final = y_raw.loc[common_index]
    dates_final = dates.loc[common_index]
    
    print(f"Forme finale après engineering: X={X_final.shape}, y={y_final.shape}")

    # >>> 3. Normalisation (Obligatoire pour le Quantique & Neural Nets)
    # Le hardware quantique et les réseaux de neurones détestent les valeurs non bornées.
    # On utilise MinMaxScaler pour tout mettre entre 0 et 1 (ou -1 et 1).
    print(">>> 3. Normalisation des données")
    
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    X_scaled = scaler_X.fit_transform(X_final)
    y_scaled = scaler_y.fit_transform(y_final)

    # >>> 4. Séparation Train / Validation (Temporelle)
    # Ne JAMAIS utiliser train_test_split avec shuffle=True sur des séries temporelles !
    # On coupe simplement les 80% premiers jours pour le train, 20% derniers pour le test.
    print(">>> 4. Split Train/Test")
    
    split_idx = int(len(X_scaled) * 0.8)
    
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
    dates_train, dates_val = dates_final[:split_idx], dates_final[split_idx:]
    
    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    return (X_train, y_train), (X_val, y_val), (scaler_X, scaler_y)

def run_advanced_pipeline(file_path='train.xlsx', forecast_horizon=1):
    # 1. Chargement initial
    print(">>> 1. Chargement des données")
    X_raw, y_raw, dates = data.load_and_prepare_data(file_path, forecast_horizon)

    # 2. Extraction de Features Techniques (NOUVEAU)
    # On utilise la fonction fournie pour extraire la "physique" du marché
    print(">>> 2. Génération des indicateurs techniques (Surface stats, Time)")
    # Cette fonction ajoute : Moyennes mobiles, Volatilité de surface, Skew, etc.
    X_tech = data.create_technical_features(X_raw, dates)
    
    print(f"Features ajoutées. Dimensions : {X_raw.shape[1]} -> {X_tech.shape[1]}")

    # 3. Création des Lags (Mémoire temporelle)
    # On applique les lags sur TOUTES les features (Prix bruts + Indicateurs techniques)
    print(">>> 3. Création de la mémoire temporelle (Lags)")
    X_lagged = data.create_lagged_features(X_tech, feature_cols=X_tech.columns, lags=[1, 2, 5])

    # 4. Alignement (Nettoyage des NaN créés par les lags/moyennes mobiles)
    common_index = X_lagged.index.intersection(y_raw.index)
    X_final = X_lagged.loc[common_index]
    y_final = y_raw.loc[common_index]
    
    # 5. Normalisation
    print(">>> 5. Normalisation MinMax")
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    X_scaled = scaler_X.fit_transform(X_final)
    y_scaled = scaler_y.fit_transform(y_final)

    # 6. Split Temporel (80/20)
    split_idx = int(len(X_scaled) * 0.8)
    
    return {
        'X_train': X_scaled[:split_idx], 'y_train': y_scaled[:split_idx],
        'X_val': X_scaled[split_idx:], 'y_val': y_scaled[split_idx:],
        'scalers': (scaler_X, scaler_y),
        'dates_val': dates.loc[common_index][split_idx:]
    }

#if __name__ == "__main__":
#    # Test du pipeline
#    (X_train, y_train), (X_val, y_val), scalers = run_data_pipeline()
#    
#    # Exemple : Affichage de la première feature sur le temps
#    plt.plot(y_train[:, 0], label="Target (Train)")
#    plt.plot(range(len(y_train), len(y_train)+len(y_val)), y_val[:, 0], label="Target (Val)")
#    plt.title("Visualisation de la séparation temporelle")
#    plt.legend()
#    plt.show()
#

if __name__ == "__main__":
    # Test rapide pour vérifier les dimensions
    data_dict = run_advanced_pipeline()
    print(f"Prêt pour l'entraînement. Input shape: {data_dict['X_train'].shape}")



  
    