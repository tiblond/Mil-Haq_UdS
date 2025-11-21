import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer  # <--- 1. AJOUT IMPORT ICI
import data  # Le fichier fourni par les organisateurs

def get_lstm_ready_data(file_path='train.xlsx', seq_length=10):
    print(">>> 1. Chargement des données brutes")
    # Charge et trie par date
    df_raw = data.load_data(file_path)
    
    # Sauvegarde des dates pour plus tard
    dates = df_raw['Date']
    
    # --- DÉBUT DU BLOC AJOUTÉ (IMPUTATION) ---
    print(">>> 1b. Imputation des données manquantes (KNN)")
    # On isole les features numériques (tout sauf la date)
    features_raw = df_raw.drop(columns=['Date'])
    
    # On utilise KNN pour remplir les trous (lignes violettes) intelligemment
    # weights='distance' donne plus d'importance aux jours très proches temporellement
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    
    # Cela retourne un numpy array sans les noms de colonnes
    features_imputed = imputer.fit_transform(features_raw)
    
    # On remet ça proprement dans un DataFrame pour la suite
    df_clean = pd.DataFrame(features_imputed, columns=features_raw.columns)
    # --- FIN DU BLOC AJOUTÉ ---
    
    print(">>> 2. Feature Engineering (Indicateurs Techniques)")
    # MODIFICATION ICI : On utilise df_clean (qui est plein) au lieu de df_raw
    X_tech = data.create_technical_features(df_clean, dates) #
    
    # Gestion des NaN créés par les moyennes mobiles (rolling windows) au tout début du fichier
    X_tech = X_tech.fillna(0)
    
    print(f"Features: {df_raw.shape[1]-1} -> {X_tech.shape[1]} (Enrichies)")

    print(">>> 3. Normalisation")
    # On normalise tout entre 0 et 1
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_tech.values)

    scaler_y = MinMaxScaler()
    y_columns = X_tech.columns[:224] 
    y_scaled = scaler_y.fit_transform(df_raw[y_columns].values)

    # On remet en DataFrame pour utiliser la fonction prepare_data de data.py
    df_ready = pd.DataFrame(X_scaled, columns=X_tech.columns)
    df_ready['Date'] = dates.reset_index(drop=True) # Nécessaire pour prepare_data

    print(f">>> 4. Création des séquences (3D pour LSTM)")
    # prepare_data gère la création des fenêtres glissantes
    # X: (Samples, Time_Steps, Features)
    # y: (Samples, Features) -> Cible à t+1
    X, y, dates_out = data.prepare_data(df_ready, forecast_horizon=1, sequence_length=seq_length) #
    
    return X, y, scaler_y

if __name__ == "__main__":
    X, y, scaler = get_lstm_ready_data()
    print(f"Données prêtes pour LSTM. Forme X: {X.shape}")