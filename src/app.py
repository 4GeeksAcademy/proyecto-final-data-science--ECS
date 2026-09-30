import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# -----------------------------
# RUTAS
# -----------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


# -----------------------------
# CARGAMOS LOS MODELOS
# -----------------------------

modelo_anchoa = joblib.load(
    MODELS_DIR / "modelo_engraulis.pkl"
)

modelo_sardina = joblib.load(
    MODELS_DIR / "modelo_sardinops.pkl"
)

variables = joblib.load(
    MODELS_DIR / "variables_modelos.pkl"
)

medianas = joblib.load(
    MODELS_DIR / "medianas_modelos.pkl"
)


# -----------------------------
# NOMBRES PARA LA APLICACIÓN
# -----------------------------

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
    "Depthm": "Profundidad (m)",
    "Bottom_D": "Profundidad del fondo (m)",
    "Wind_Spd": "Velocidad del viento (nudos)"
}


# -----------------------------
# TÍTULO
# -----------------------------

st.title("Probabilidad estimada de presencia")

st.write(
    "Introduce las condiciones del muestreo para estimar "
    "la probabilidad de presencia de anchoa y sardina."
)


# -----------------------------
# DOS COLUMNAS
# -----------------------------

col1, col2 = st.columns([1, 1])


# -----------------------------
# DATOS DEL USUARIO
# -----------------------------

valores = {}

with col1:

    st.subheader("Condiciones del muestreo")

    for variable in variables:

        nombre = nombres_bonitos.get(variable, variable)
        valor_inicial = medianas.get(variable, 0)

        if variable == "Year":

            valores[variable] = st.number_input(
                nombre,
                value=int(round(valor_inicial)),
                step=1,
                format="%d",
                key=f"campo_{variable}"
            )

        elif variable == "Month":

            valores[variable] = st.number_input(
                nombre,
                min_value=1,
                max_value=12,
                value=int(round(valor_inicial)),
                step=1,
                format="%d",
                key=f"campo_{variable}"
            )

        elif variable in ["Depthm", "Bottom_D"]:

            valores[variable] = st.number_input(
                nombre,
                value=int(round(valor_inicial)),
                step=1,
                format="%d",
                key=f"campo_{variable}"
            )

        else:

            valores[variable] = st.number_input(
                nombre,
                value=float(valor_inicial),
                format="%.2f",
                key=f"campo_{variable}"
            )


# -----------------------------
# MAPA
# -----------------------------

with col2:

    st.subheader("Mapa de California")

    latitud = valores.get("Lat_Dec", 34.0)
    longitud = valores.get("Lon_Dec", -120.0)

    punto_mapa = pd.DataFrame({
        "lat": [latitud],
        "lon": [longitud]
    })

    st.map(
        punto_mapa,
        latitude="lat",
        longitude="lon",
        zoom=5
    )


# -----------------------------
# PREDICCIÓN
# -----------------------------

st.divider()

if st.button("Calcular probabilidad de cada especie"):

    entrada = pd.DataFrame(
        [[valores[v] for v in variables]],
        columns=variables
    )

    prob_anchoa = modelo_anchoa.predict_proba(
        entrada
    )[0][1]

    prob_sardina = modelo_sardina.predict_proba(
        entrada
    )[0][1]

    porcentaje_anchoa = prob_anchoa * 100
    porcentaje_sardina = prob_sardina * 100


    st.subheader("Probabilidad estimada de presencia")

    st.write(
        f"Anchoa: {porcentaje_anchoa:.1f}%"
    )

    st.progress(prob_anchoa)


    st.write(
        f"Sardina: {porcentaje_sardina:.1f}%"
    )

    st.progress(prob_sardina)


    if porcentaje_anchoa > porcentaje_sardina:

        st.success(
            "Según el modelo, la anchoa presenta "
            "mayor probabilidad estimada de presencia."
        )

    else:

        st.success(
            "Según el modelo, la sardina presenta "
            "mayor probabilidad estimada de presencia."
        )


    st.info(
        "Las probabilidades son estimaciones realizadas por los modelos "
        "a partir de los datos históricos utilizados durante su entrenamiento."
    )