"""
En este archivo hay creacion de variables Dummies, escalado y normalizacion de variables.
"""
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parents[2]

def build_preprocessor(categorical_features, numerical_features):

    categorical_transformer = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False
    )

    numerical_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numerical_transformer, numerical_features)
        ]
    )

    return preprocessor


def run_features_pipeline(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica escalado y variables dummies al DataFrame limpio del ETL.
    
    Parámetros:
    df_clean: DataFrame limpio generado por el ETL
    
    Retorna:
    pd.DataFrame con los datos procesados (escalados y con dummies)
    """
    
    # Todas las features excepto el target
    TARGET = "Consumo_Television"
    
    categorical_features = [
        "Region", "Genero", "Grupo_Edad", "Nivel_Socioeconomico", 
        "Estudios_Alcanzados", "Trabajo", "Consumo_Plataformas_Digitales", 
        "Consumo_Teatro", "Consumo_Musica"
    ]
    
    numerical_features = []
    
    # Separar las features del DataFrame (sin el target)
    X = df_clean[categorical_features].copy()
    
    # Crear y ajustar el preprocessor
    preprocessor = build_preprocessor(categorical_features, numerical_features)
    
    # Aplicar transformaciones
    X_transformed = preprocessor.fit_transform(X)
    
    # Obtener nombres de columnas después de OneHotEncoder
    categorical_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
    all_feature_names = categorical_names
    
    # Convertir a DataFrame con nombres de columnas apropiados
    df_features = pd.DataFrame(X_transformed, columns=all_feature_names)
    
    return df_features





