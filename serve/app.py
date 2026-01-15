from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import joblib
import numpy as np
import sys

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from output.models.train import train
from output.models.evaluate import evaluate
from pipelines.consumos_culturales_data_science.etl import run_etl
from output.features.features import run_features_pipeline

app = FastAPI(
    title="Data Science API",
    description="API para predicción de consumo de televisión",
    version="1.0.0"
)

# Ruta del modelo entrenado
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "model_tv.pkl"

# Cargar modelo al iniciar
model = None

@app.on_event("startup")
async def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print("✓ Modelo cargado exitosamente")
    else:
        print("⚠ Modelo no encontrado. Entrena el modelo primero con: python main.py")


# Modelos Pydantic para validación
class PredictionInput(BaseModel):
    features: list[float]
    class Config:
        example = {
            "features": [1.0, 2.0, 3.0, 4.0]  # Ajusta según tus features
        }

class PredictionOutput(BaseModel):
    prediction: int
    probability: float


# Endpoints
@app.get("/")
async def root():
    return {
        "message": "Bienvenido a la API de Data Science",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no cargado. Entrena el modelo primero."
        )
    
    try:
        # Convertir features a array numpy
        X = np.array(input_data.features).reshape(1, -1)
        
        # Hacer predicción
        prediction = model.predict(X)[0]
        probability = float(model.predict_proba(X).max())
        
        return PredictionOutput(
            prediction=int(prediction),
            probability=probability
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error en la predicción: {str(e)}"
        )


@app.post("/train")
async def train_model():
    """Entrena un modelo nuevo"""
    try:
        from output.models.train import train
        model_trained = train()
        return {"message": "Modelo entrenado exitosamente"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al entrenar: {str(e)}"
        )


@app.get("/metrics")
async def get_metrics():
    """Obtiene métricas del modelo evaluado"""
    try:
        from output.models.evaluate import evaluate
        metrics = evaluate()
        return metrics
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al evaluar: {str(e)}"
        )
