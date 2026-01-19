"""
En este archivo hay creacion de variables Dummies, escalado y normalizacion de variables.
Soporta dos métodos: OneHotEncoder y CatPCA para experimentación.
"""
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import prince

BASE_DIR = Path(__file__).resolve().parents[2]

def build_preprocessor(categorical_features, numerical_features, method="onehot", n_components=None):
    """
    Construye un preprocesador para transformar features categóricas y numéricas.
    
    Parámetros:
    categorical_features: lista de nombres de features categóricas
    numerical_features: lista de nombres de features numéricas
    method: "onehot" para OneHotEncoder o "catpca" para CatPCA
    n_components: número de componentes para CatPCA
    
    Retorna:
    preprocessor: ColumnTransformer configurado
    """
    
    if method.lower() == "catpca":
        # Usar MCA (Multiple Correspondence Analysis) para datos categóricos
        if n_components is None:
            n_components = min(len(categorical_features) - 1, 5)
        categorical_transformer = prince.MCA(
            n_components=n_components,
            n_iter=3,
            random_state=42
        )
    else:
        # Usar OneHotEncoder (default)
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


def run_features_pipeline(df_clean: pd.DataFrame, method: str = "onehot", n_components: int = None) -> pd.DataFrame:
    """
    Aplica escalado y variables dummies (o CatPCA) al DataFrame limpio del ETL.
    
    Parámetros:
    df_clean: DataFrame limpio generado por el ETL
    method: "onehot" para OneHotEncoder o "catpca" para CatPCA
    n_components: número de componentes para CatPCA
    
    Retorna:
    pd.DataFrame con los datos procesados
    """
    
    categorical_features = [
        "Region", "Genero", "Grupo_Edad", "Nivel_Socioeconomico", 
        "Estudios_Alcanzados", "Trabajo", "Consumo_Plataformas_Digitales", 
        "Consumo_Teatro", "Consumo_Musica"
    ]
    
    numerical_features = []
    
    # Separar las features del DataFrame (sin el target)
    X = df_clean[categorical_features].copy()
    
    # Crear y ajustar el preprocessor
    preprocessor = build_preprocessor(
        categorical_features, 
        numerical_features, 
        method=method,
        n_components=n_components
    )
    
    # Aplicar transformaciones
    X_transformed = preprocessor.fit_transform(X)
    
    # Obtener nombres de columnas según el método utilizado
    if method.lower() == "catpca":
        if isinstance(X_transformed, pd.DataFrame):
            all_feature_names = X_transformed.columns.tolist()
        else:
            n_cols = X_transformed.shape[1]
            all_feature_names = [f"catpca_{i}" for i in range(n_cols)]
    else:
        # OneHotEncoder
        categorical_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
        all_feature_names = categorical_names
    
    # Convertir a DataFrame con nombres de columnas apropiados
    df_features = pd.DataFrame(X_transformed, columns=all_feature_names)
    
    return df_features





