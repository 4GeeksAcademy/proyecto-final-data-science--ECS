import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# Rutas
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


# Cargamos los dos modelos
modelo_engraulis = joblib.load(
    MODELS_DIR / "modelo_engraulis.pkl"
)

modelo_sardinops = joblib.load(
    MODELS_DIR / "modelo_sardinops.pkl"
)

variables = joblib.load(
    MODELS_DIR / "variables_modelos.pkl"
)

medianas = joblib.load(
    MODELS_DIR / "medianas_modelos.pkl"
)


# Nombres más claros para mostrar
nombres_bonitos = {
    "Year": "Año",
    "Month": "Mes",
    "Lat_Dec": "Latitud",
    "Lon_Dec": "Longitud",
    "T_degC": "Temperatura del agua (°C)",
    "Salnty": "Salinidad",
    "STheta": "Densidad potencial",
    "O2ml_L": "Oxígeno disuelto",
    "O2Sat": "Saturación de oxígeno",
    "Depthm": "Profundidad",
    "Bottom_D": "Profundidad del fondo",
    "Wind_Spd": "Velocidad del viento"
}


st.title("Probabilidad estimada de presencia")

st.write(
    "Introduce las condiciones del muestreo para estimar "
    "la presencia de Engraulis mordax y Sardinops sagax."
)


# Campos de entrada
valores = []

for variable in variables:

    nombre = nombres_bonitos.get(
        variable,
        variable
    )

    valor_inicial = float(
        medianas.get(variable, 0)
    )

    valor = st.number_input(
        nombre,
        value=valor_inicial,
        key=f"campo_{variable}"
    )

    valores.append(valor)


# Predicción
if st.button("Calcular probabilidad"):

    entrada = pd.DataFrame(
        [valores],
        columns=variables
    )

    # Probabilidad de presencia = clase 1
    prob_engraulis = modelo_engraulis.predict_proba(
        entrada
    )[0][1]

    prob_sardinops = modelo_sardinops.predict_proba(
        entrada
    )[0][1]

    porcentaje_engraulis = prob_engraulis * 100
    porcentaje_sardinops = prob_sardinops * 100


    st.subheader("Probabilidad estimada de presencia")

    st.write(
        f"Engraulis mordax: {porcentaje_engraulis:.1f}%"
    )

    st.progress(prob_engraulis)


    st.write(
        f"Sardinops sagax: {porcentaje_sardinops:.1f}%"
    )

    st.progress(prob_sardinops)


    st.info(
        "Estas probabilidades son estimaciones del modelo "
        "a partir de los datos históricos utilizados para entrenarlo."
    )

    if porcentaje_sardinops > porcentaje_engraulis:
        st.write(
            "Según el modelo, Sardinops sagax presenta mayor "
            "probabilidad estimada de presencia en estas condiciones."
        )
    else:
        st.write(
            "Según el modelo, Engraulis mordax presenta mayor "
            "probabilidad estimada de presencia en estas condiciones."
        )