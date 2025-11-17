import optuna
import mlflow
import mlflow.sklearn
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from etl import load_raw, clean_data
from feature_engineer import crear_materializado_30d
from train import Train


# =========================
# OBJETIVO PARA OPTUNA
# =========================
def objective(trial):

    # Hiperparámetros a optimizar
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }

    model = GradientBoostingClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        random_state=42
    )

    # Entrenamiento
    trainer = Train(df_global, target_column, model)
    pipeline, X_train, X_test, y_train, y_test = trainer.train()

    # Métrica objetivo
    probs = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    return auc


if __name__ == "__main__":

    # =========================
    # 1. Cargar los datos
    # =========================
    df_global = load_raw("datos/df_limpio_2023.xlsx")
    df_global = clean_data(df_global)
    df_global = crear_materializado_30d(df_global)

    target_column = "materializado"

    # =========================
    # 2. Crear estudio Optuna
    # =========================
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print("\nMejores hiperparámetros:", study.best_params)
    print("Mejor AUC:", study.best_value)

    # =========================
    # 3. Entrenar modelo final con los mejores hiperparámetros
    # =========================
    best_params = study.best_params

    best_model = GradientBoostingClassifier(
        **best_params,
        random_state=42
    )

    trainer = Train(df_global, target_column, best_model)
    best_pipeline, *_ = trainer.train()

    # =========================
    # 4. Registrar en MLflow
    # =========================
    with mlflow.start_run():

        mlflow.log_params(best_params)
        mlflow.sklearn.log_model(best_pipeline, "modelo_optuna")

        # Capturar el run_id
        print(f"Run ID: {mlflow.active_run().info.run_id}")

    # =========================
    # 5. Guardar modelo local
    # =========================
    joblib.dump(best_pipeline, "modelos/modelo_optuna.pkl")
    