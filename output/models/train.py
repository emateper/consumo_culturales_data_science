from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from pipelines.consumos_culturales_data_science.etl import run_etl
from output.features.features import run_features_pipeline

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "model_tv.pkl"

# Asegurar que la carpeta models existe
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def train():
    # Ejecutar ETL para limpiar datos
    print("Ejecutando ETL...")
    df_clean = run_etl()

    # Ejecutar Features Pipeline con los datos limpios
    print("Ejecutando Features Pipeline...")
    df_features = run_features_pipeline(df_clean)

    TARGET = "Consumo_Television"
    
    # X son todas las columnas del DataFrame de features (ya escaladas y con dummies)
    X = df_features
    
    # y es el target del DataFrame limpio
    y = df_clean[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Entrenar modelo (sin preprocessor, pues ya fueron aplicadas las transformaciones)
    model_tv = RandomForestClassifier(random_state=42)
    model_tv.fit(X_train, y_train)

    y_pred = model_tv.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model_tv, MODEL_PATH)
    joblib.dump(
        {"X_test": X_test, "y_test": y_test},
        BASE_DIR / "models" / "test_data.pkl"
    )

    return model_tv


if __name__ == "__main__":
    train()
