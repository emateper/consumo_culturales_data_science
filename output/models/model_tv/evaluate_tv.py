import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import json

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models_pkl_tv" / "model_tv.pkl"
TEST_PATH = BASE_DIR / "models_pkl_tv" / "test_data_tv.pkl"


def evaluate_tv():
    """
    Evalúa el modelo entrenado y retorna un diccionario con todas las métricas.
    
    Retorna:
    dict: Diccionario con las siguientes métricas:
        - accuracy: Precisión general del modelo
        - precision: Precisión ponderada
        - recall: Cobertura ponderada
        - f1: F1-score ponderado
        - classification_report: Reporte de clasificación por clase
        - confusion_matrix: Matriz de confusión
    """
    model = joblib.load(MODEL_PATH)
    test_data = joblib.load(TEST_PATH)

    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    y_pred = model.predict(X_test)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Obtener reporte de clasificación como diccionario
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    
    # Matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    # Armar diccionario de resultados
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "classification_report": classification_rep,
        "confusion_matrix": conf_matrix
    }

    return metrics


def print_metrics(metrics: dict):
    """
    Imprime las métricas de forma legible.
    
    Parámetros:
    metrics: Diccionario de métricas retornado por evaluate()
    """
    print("=" * 50)
    print("MÉTRICAS DEL MODELO")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print("\nReporte de Clasificación:")
    print(json.dumps(metrics['classification_report'], indent=2))
    print("\nMatriz de Confusión:")
    print(metrics['confusion_matrix'])
    print("=" * 50)


if __name__ == "__main__":
    metrics = evaluate_tv()
    print_metrics(metrics)
