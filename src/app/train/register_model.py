import mlflow
import mlflow.sklearn
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def register_model(model_uri, model_name, stage=None):
    
    logger.info(f"Registrando modelo: {model_name}")
    logger.info(f"MLflow model_uri: {model_uri}")

    # Registrar el modelo
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    version = result.version
    logger.info(f"Modelo registrado: {model_name}, versión: {version}")

   