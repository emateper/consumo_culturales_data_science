from output.models.train import train
from output.models.evaluate import evaluate, print_metrics
import argparse
import sys

def main():
    print("\n" + "="*50)
    print("INICIANDO PIPELINE DE ENTRENAMIENTO")
    print("="*50 + "\n")
    
    # Entrenar modelo
    print("1. Entrenando modelo...")
    model = train()
    print("   ✓ Modelo entrenado exitosamente\n")
    
    # Evaluar modelo
    print("2. Evaluando modelo...")
    metrics = evaluate()
    print("   ✓ Evaluación completada\n")
    
    # Mostrar métricas
    print_metrics(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Science Pipeline")
    parser.add_argument(
        "--api",
        action="store_true",
        help="Inicia el servidor FastAPI (uvicorn serve.app:app)"
    )
    
    args = parser.parse_args()
    
    if args.api:
        print("\nIniciando servidor FastAPI...")
        print("Accede a http://localhost:8000/docs para ver la documentación\n")
        import uvicorn
        uvicorn.run("serve.app:app", host="0.0.0.0", port=8000, reload=True)
    else:
        main()