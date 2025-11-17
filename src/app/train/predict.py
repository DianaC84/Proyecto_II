import joblib
import pandas as pd
from feature_engineer import crear_materializado_30d

def cargar_modelo(path):
    return joblib.load(path)

def preparar_nuevos_datos(df, columnas_entrenamiento):
    df = crear_materializado_30d(df)

    # asegurar mismo orden y mismas columnas
    df = df[columnas_entrenamiento]

    return df

def predecir(df, model):
    pred_prob = model.predict_proba(df)[:, 1]
    pred_class = model.predict(df)
    df["probabilidad_materializado"] = pred_prob
    df["prediccion"] = pred_class
    return df

if __name__ == "__main__":
    modelo = cargar_modelo("modelos/modelo_riesgos.pkl")
    
    # obtener columnas del pipeline entrenado
    columnas_entrenamiento = modelo.feature_names_in_

    df_nuevo = pd.read_excel("datos/datos_nuevos.xlsx")
    df_nuevo = preparar_nuevos_datos(df_nuevo, columnas_entrenamiento)

    df_pred = predecir(df_nuevo, modelo)
    df_pred.to_excel("datos/predicciones.xlsx", index=False)

