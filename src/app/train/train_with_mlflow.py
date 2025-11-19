import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split

# Importar funciones desde tu train.py
from train import (
    cargar_dataset,
    preparar_datos,
    entrenar_modelo,
    evaluar_modelo
)

# ================================
# CONFIGURAR MLflow
# ================================

mlflow.set_experiment("GradientBoosting_Materializado")

print("\n🚀 Iniciando entrenamiento con MLflow...\n")

# ================================
# 1. Cargar dataset con features
# ================================

df = cargar_dataset("datos_features.xlsx")

# ================================
# 2. Preparar datos
# ================================

X, y = preparar_datos(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("✔ Datos divididos correctamente.")

# ================================
# 3. MLflow logging
# ================================

with mlflow.start_run():

    # 3.1 Entrenar modelo usando TU función original
    modelo = entrenar_modelo(X_train, y_train)

    # 3.2 Evaluar modelo
    umbral_optimo, auc, f1, acc = evaluar_modelo(modelo, X_test, y_test)

    # 3.3 Loguear parámetros
    mlflow.log_param("modelo", "GradientBoostingClassifier")
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_param("umbral_optimo", umbral_optimo)

    # 3.4 Loguear métricas
    mlflow.log_metric("AUC", auc)
    mlflow.log_metric("F1", f1)
    mlflow.log_metric("Accuracy", acc)

    # 3.5 Loguear modelo en MLflow
    mlflow.sklearn.log_model(modelo, artifact_path="modelo_gradient_boosting")

print("\n🎉 Entrenamiento con MLflow finalizado.")