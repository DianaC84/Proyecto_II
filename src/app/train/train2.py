import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, precision_recall_curve
)

# ---------------------------------------------------
# 1. FUNCIONES
# ---------------------------------------------------

def cargar_dataset(ruta: str) -> pd.DataFrame:
    print("📄 Cargando dataset...")
    df = pd.read_excel(ruta)
    print(f"✔ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas.")
    return df


def preparar_datos(df: pd.DataFrame):
    columnas = [
        "Causa_DS",
        "AgenteGenerador_DS",
        "Severidad_NUM",
        "año",
        "valor_acumulado"
    ]

    X = df[columnas]
    y = df["materializado_30d"]

    print("\nVariables usadas para el modelo:")
    print(columnas)

    return X, y


def calcular_umbral_optimo(y_true, probs):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    idx = np.argmax(f1_scores)
    return thresholds[idx]


def evaluar(modelo, X, y, nombre_conjunto):
    print(f"\n----- Evaluando {nombre_conjunto} -----")

    probs = modelo.predict_proba(X)[:, 1]
    umbral = calcular_umbral_optimo(y, probs)
    y_pred = (probs >= umbral).astype(int)

    auc = roc_auc_score(y, probs)
    f1 = f1_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    print(f"AUC: {auc:.4f}")
    print(f"F1 : {f1:.4f}")
    print(f"ACC: {acc:.4f}")
    print(f"Matriz Confusión:\n{cm}")

    return auc, f1, acc


def detectar_overfitting(auc_train, auc_test):
    print("\n----- ANÁLISIS DE OVERFITTING -----")

    if auc_train - auc_test > 0.07:
        print("⚠️ OVERFITTING DETECTADO: El modelo rinde mucho mejor en train que en test.")
    elif auc_train - auc_test > 0.03:
        print("⚠️ Posible overfitting moderado.")
    else:
        print("✔ No hay señales fuertes de overfitting.")
        

# ---------------------------------------------------
# 2. MAIN
# ---------------------------------------------------

if __name__ == "__main__":

    # 1. Cargar data
    df = cargar_dataset("datos_features.xlsx")

    # 2. Preparar X y y
    X, y = preparar_datos(df)

    # 3. Split idéntico al train.py
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Cargar modelo entrenado
    print("\n📦 Cargando modelo entrenado...")
    modelo = joblib.load("modelo_gradientboosting_optimo.pkl")
    print("✔ Modelo cargado.")

    # 5. Evaluación en TRAIN y TEST
    auc_train, f1_train, acc_train = evaluar(modelo, X_train, y_train, "TRAIN")
    auc_test, f1_test, acc_test = evaluar(modelo, X_test, y_test, "TEST")

    # 6. Verificar overfitting
    detectar_overfitting(auc_train, auc_test)