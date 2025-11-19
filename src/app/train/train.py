import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, confusion_matrix,
    classification_report, f1_score, accuracy_score
)
from sklearn.ensemble import GradientBoostingClassifier

# 1. CARGAR DATASET

def cargar_dataset(ruta: str) -> pd.DataFrame:
    print("Cargando dataset final con features...")
    df = pd.read_excel(ruta)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas.")
    return df


# 2. SELECCIONAR FEATURES DEFINIDOS (X_v2)

def preparar_datos(df: pd.DataFrame):

    columnas_features = [
        "Causa_DS",
        "AgenteGenerador_DS",
        "Severidad_NUM",
        "año",
        "valor_acumulado"
    ]

    print("\nUsando las siguientes features:")
    print(columnas_features)

    X = df[columnas_features]
    y = df["materializado_30d"]

    return X, y


# 3. ENTRENAR MODELO

def entrenar_modelo(X_train, y_train):

    # ⭐ Hiperparámetros óptimos encontrados con Optuna
    best_params = {
        'n_estimators': 258,
        'learning_rate': 0.10233634007162776,
        'max_depth': 5,
        'subsample': 0.9472556968322051,
        'min_samples_split': 16,
        'min_samples_leaf': 8
    }

    print("\nEntrenando modelo GradientBoosting con hiperparámetros óptimos...")
    
    modelo = GradientBoostingClassifier(
        **best_params,
        random_state=42
    )

    modelo.fit(X_train, y_train)
    print("Entrenamiento finalizado.")
    return modelo


# 4. CÁLCULO DEL UMBRAL ÓPTIMO

def obtener_umbral_optimo(y_true, probs):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    idx = np.argmax(f1_scores)
    umbral = thresholds[idx]
    return umbral, f1_scores[idx]


# 5. EVALUACIÓN

def evaluar_modelo(modelo, X_test, y_test):
    probs = modelo.predict_proba(X_test)[:, 1]

    umbral_optimo, best_f1 = obtener_umbral_optimo(y_test, probs)

    y_pred = (probs >= umbral_optimo).astype(int)

    print("\n========== RESULTADOS ==========\n")
    print(f"Umbral óptimo (basado en F1): {umbral_optimo:.3f}")
    print("\nMatriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, digits=3))

    auc = roc_auc_score(y_test, probs)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("\n📊 MÉTRICAS FINALES:")
    print(f"AUC: {auc:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"Accuracy: {acc:.3f}")

    return umbral_optimo, auc, f1, acc

# 6. MAIN

if __name__ == "__main__":

    # 1) Cargar datos procesados
    df = cargar_dataset("datos_features.xlsx")

    # 2) Preparar dataset 
    X, y = preparar_datos(df)

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Entrenar modelo
    modelo = entrenar_modelo(X_train, y_train)

    # 5) Evaluación
    umbral_optimo, auc, f1, acc = evaluar_modelo(modelo, X_test, y_test)

    # 6) Guardar modelo y umbral
    joblib.dump(modelo, "modelo_gradientboosting_optimo.pkl")
    print("\n💾 Modelo guardado como 'modelo_gradientboosting_optimo.pkl'")

    with open("umbral_optimo.txt", "w") as f:
        f.write(str(umbral_optimo))

    print("💾 Umbral óptimo guardado en 'umbral_optimo.txt'")