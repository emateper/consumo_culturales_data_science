import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from serve.app import app

client = TestClient(app)


def test_root():
    """Test del endpoint raíz"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check():
    """Test del health check"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"


def test_predict_without_model():
    """Test de predicción sin modelo cargado"""
    response = client.post("/predict", json={"features": [1.0, 2.0, 3.0, 4.0]})
    # Si no hay modelo, debería retornar 503 o procesar según la lógica
    assert response.status_code in [503, 400, 500]


def test_predict_invalid_input():
    """Test con input inválido"""
    response = client.post("/predict", json={"invalid": "data"})
    assert response.status_code == 422  # Validation error


def test_metrics_endpoint():
    """Test del endpoint de métricas"""
    response = client.get("/metrics")
    assert response.status_code in [200, 500]  # 500 si no hay modelo


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
