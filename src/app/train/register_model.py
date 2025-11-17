import mlflow
from mlflow.tracking import MlflowClient

def register_model(run_id, model_name, artifact_path):
    client = MlflowClient()
    
    # Crear modelo registrado si no existe
    try:
        client.create_registered_model(model_name)
        print(f"Modelo registrado creado: {model_name}")
    except Exception as e:
        print(f"El modelo ya existe: {model_name}")

    # Crear una nueva versión del modelo
    model_version = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/{artifact_path}",
        run_id=run_id
    )
    print(f"Versión de modelo creada: {model_version.version}")

if __name__ == "__main__":
    # --- PARA MLflow ---
    # run_id = "fdb3e108350447ed9a3eccca4ad92900"
    # artifact_path = "modelo_riesgos"

    # --- PARA OPTUNA ---
    run_id = "973a34f3222649549ade2414b38876a3"
    artifact_path = "modelo_optuna"

    model_name = "modelo_riesgos_gradientboosting"

    register_model(run_id, model_name, artifact_path)


