import pandas as pd
from sklearn.impute import KNNImputer

def fill_purple_rows(df):
    print(">>> Début de l'imputation (Nettoyage des NaN)...")
    
    # 1. Séparer la date (on ne l'impute pas)
    dates = df['Date']
    features = df.drop(columns=['Date'])
    
    # 2. Utiliser KNN Imputer
    # Si une valeur manque pour une date donnée, il regarde les jours similaires
    # et les corrélations entre colonnes voisines.
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    
    features_imputed = imputer.fit_transform(features)
    
    # 3. Reconstruire le DataFrame
    df_clean = pd.DataFrame(features_imputed, columns=features.columns)
    df_clean['Date'] = dates.reset_index(drop=True)
    
    print(">>> Imputation terminée. Plus de NaN.")
    return df_clean