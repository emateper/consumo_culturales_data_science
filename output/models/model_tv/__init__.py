from output.features.features import run_features_pipeline, build_preprocessor
from .train_tv import train_tv
from .evaluate_tv import evaluate_tv, print_metrics

# Variable global para almacenar parámetros de entrenamiento
_training_params = {"method": "onehot", "n_components": None}

def set_training_params(method: str = "onehot", n_components: int = None):
    """Establece los parámetros para el entrenamiento."""
    global _training_params
    _training_params = {"method": method, "n_components": n_components}

def train_tv():
    """Entrena el modelo con los parámetros establecidos."""
    return train_tv(**_training_params)

__all__ = [
    'run_features_pipeline',
    'build_preprocessor',
    'train_tv',
    'train_tv',
    'evaluate_tv',
    'print_metrics',
    'set_training_params'
]
