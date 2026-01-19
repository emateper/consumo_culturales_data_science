from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from pipelines.consumos_culturales_data_science.etl import run_etl
from output.features.features import run_features_pipeline

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models_pkl_tv" / "model_tv.pkl"

# Asegurar que la carpeta models_pkl existe
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


# Configuración global
TRAINING_PARAMS = {
    "method": "onehot",
    "n_components": None
}

def set_training_params(method: str = "onehot", n_components: int = None):
    global TRAINING_PARAMS

    if method not in ["onehot", "catpca"]:
        raise ValueError("method debe ser 'onehot' o 'catpca'")

    TRAINING_PARAMS["method"] = method
    TRAINING_PARAMS["n_components"] = n_components


def train_tv():
    method = TRAINING_PARAMS["method"]
    n_components = TRAINING_PARAMS["n_components"]

    print("Ejecutando ETL...")
    df_clean = run_etl()

    print(f"Ejecutando Features Pipeline con método: {method}...")
    df_features = run_features_pipeline(df_clean, method=method, n_components=n_components)

    TARGET = "Consumo_Television"

    X = df_features
    y = df_clean[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_tv = RandomForestClassifier(random_state=42)
    model_tv.fit(X_train, y_train)

    y_pred = model_tv.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model_tv, MODEL_PATH)
    joblib.dump(
        {"X_test": X_test, "y_test": y_test},
        BASE_DIR / "models_pkl_tv" / "test_data_tv.pkl"
    )

    return model_tv
