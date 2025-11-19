import mlflow
import mlflow.sklearn
import joblib
import argparse
import os


def registrar_modelo(ruta_modelo: str, nombre_modelo: str):
    print("\n=== REGISTRO DE MODELO EN MLFLOW ===")

    # 1. Verificar que el archivo exista
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(
            f"❌ ERROR: No se encontró el archivo del modelo en la ruta: {ruta_modelo}"
        )

    print(f"✔ Cargando modelo desde: {ruta_modelo}")
    modelo = joblib.load(ruta_modelo)

    # 2. Configurar experimento
    experiment_name = "riesgos_gradient_boosting"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="RegistrarModelo"):
        print("✔ Registrando modelo en MLflow...")

        mlflow.sklearn.log_model(
            sk_model=modelo,
            artifact_path="modelo_gradientboosting",
            registered_model_name=nombre_modelo
        )

        print(f"\n📦 Modelo registrado en el Model Registry como: {nombre_modelo}")
        print("🎉 ¡Registro completado exitosamente!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Registrar modelo en MLflow")

    parser.add_argument(
        "--ruta_modelo",
        type=str,
        default="modelo_gradientboosting_optimo.pkl",   # ← Nombre correcto
        help="Ruta del archivo .pkl del modelo a registrar"
    )

    parser.add_argument(
        "--nombre_modelo",
        type=str,
        default="riesgos_gradientboosting_optimo",
        help="Nombre del modelo en el MLflow Model Registry"
    )

    args = parser.parse_args()

    registrar_modelo(args.ruta_modelo, args.nombre_modelo)