from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from pipelines.consumos_culturales_data_science.etl import run_etl
from output.features.features import run_features_pipeline

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models_pkl_teatro" / "model_teatro.pkl"

# Asegurar que la carpeta models_pkl existe
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


# Configuración global
TRAINING_PARAMS_TEATRO = {
    "method": "onehot",
    "n_components": None
}


def set_training_params_teatro(method: str = "onehot", n_components: int = None):
    global TRAINING_PARAMS_TEATRO

    if method not in ["onehot", "catpca"]:
        raise ValueError("method debe ser 'onehot' o 'catpca'")

    TRAINING_PARAMS_TEATRO["method"] = method
    TRAINING_PARAMS_TEATRO["n_components"] = n_components


def train_teatro():
    method = TRAINING_PARAMS_TEATRO["method"]
    n_components = TRAINING_PARAMS_TEATRO["n_components"]

    print("Ejecutando ETL...")
    df_clean = run_etl()

    print(f"Ejecutando Features Pipeline con método: {method}...")
    df_features = run_features_pipeline(df_clean, method=method, n_components=n_components)

    TARGET = "Consumo_Teatro"
    
    X = df_features.drop(columns=[TARGET], errors="ignore")

    y = df_clean[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_teatro = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=10,
    min_samples_split=10,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
    model_teatro.fit(X_train, y_train)

    y_pred = model_teatro.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model_teatro, MODEL_PATH)
    joblib.dump(
        {"X_test": X_test, "y_test": y_test},
        BASE_DIR / "models_pkl_teatro" / "test_data_teatro.pkl"
    )

    return model_teatro

if __name__ == "__main__":
    train_teatro()


