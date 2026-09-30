import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Clasificador de especies marinas",
    page_icon="🐟",
    layout="wide"
)


st.title("🐟 Clasificador de especies marinas")

st.write(
    "Aplicación de Machine Learning para diferenciar "
    "Engraulis mordax y Sardinops sagax."
)

st.write(
    "Introduce las condiciones ambientales y oceanográficas "
    "para realizar una predicción."
)


modelo = joblib.load(
    "notebooks/individuales/modelo_xgboost.pkl"
)

st.success("Modelo XGBoost cargado correctamente.")


st.subheader("📊 Datos de entrada")


columna1, columna2 = st.columns(2)


with columna1:

    Year = st.number_input(
        "Año",
        min_value=1900,
        max_value=2100,
        value=2019
    )

    Month = st.number_input(
        "Mes",
        min_value=1,
        max_value=12,
        value=6
    )

    Lat_Dec = st.number_input(
        "Latitud",
        value=30.0
    )

    Lon_Dec = st.number_input(
        "Longitud",
        value=-120.0
    )

    T_degC = st.number_input(
        "Temperatura (°C)",
        value=15.0
    )

    Salnty = st.number_input(
        "Salinidad",
        value=33.0
    )

    STheta = st.number_input(
        "Densidad potencial (STheta)",
        value=25.0
    )

    O2ml_L = st.number_input(
        "Oxígeno disuelto (ml/L)",
        value=5.0
    )

    O2Sat = st.number_input(
        "Saturación de oxígeno (%)",
        value=100.0
    )

    Bottom_D = st.number_input(
        "Profundidad del fondo",
        value=100.0
    )


with columna2:

    Distance = st.number_input(
        "Distancia",
        value=100.0
    )

    ChlorA = st.number_input(
        "Clorofila",
        value=0.5
    )

    Phaeop = st.number_input(
        "Phaeop",
        value=0.1
    )

    PO4uM = st.number_input(
        "Fosfato (PO4uM)",
        value=1.0
    )

    SiO3uM = st.number_input(
        "Silicato (SiO3uM)",
        value=5.0
    )

    NO2uM = st.number_input(
        "Nitrito (NO2uM)",
        value=0.1
    )

    NO3uM = st.number_input(
        "Nitrato (NO3uM)",
        value=5.0
    )

    IntChl = st.number_input(
        "Clorofila integrada",
        value=1.0
    )

    Wind_Spd = st.number_input(
        "Velocidad del viento",
        value=5.0
    )

    Wave_Ht = st.number_input(
        "Altura de las olas",
        value=1.0
    )


st.divider()


if st.button(
    "🔍 Realizar predicción",
    use_container_width=True
):

    datos = pd.DataFrame([{
        "Year": Year,
        "Month": Month,
        "Lat_Dec": Lat_Dec,
        "Lon_Dec": Lon_Dec,
        "T_degC": T_degC,
        "Salnty": Salnty,
        "STheta": STheta,
        "O2ml_L": O2ml_L,
        "O2Sat": O2Sat,
        "Bottom_D": Bottom_D,
        "Distance": Distance,
        "ChlorA": ChlorA,
        "Phaeop": Phaeop,
        "PO4uM": PO4uM,
        "SiO3uM": SiO3uM,
        "NO2uM": NO2uM,
        "NO3uM": NO3uM,
        "IntChl": IntChl,
        "Wind_Spd": Wind_Spd,
        "Wave_Ht": Wave_Ht
    }])


    prediccion = modelo.predict(datos)[0]

    probabilidades = modelo.predict_proba(datos)[0]

    probabilidad_anchoa = probabilidades[0]

    probabilidad_sardina = probabilidades[1]


    st.subheader("🐟 Resultado de la predicción")


    if prediccion == 1:

        st.success("Sardinops sagax")

    else:

        st.info("Engraulis mordax")


    st.write(
        f"Probabilidad de Engraulis mordax: "
        f"{probabilidad_anchoa:.2%}"
    )

    st.write(
        f"Probabilidad de Sardinops sagax: "
        f"{probabilidad_sardina:.2%}"
    )


    st.subheader("📊 Probabilidad de cada especie")


    grafico = pd.DataFrame({
        "Especie": [
            "Engraulis mordax",
            "Sardinops sagax"
        ],
        "Probabilidad": [
            probabilidad_anchoa,
            probabilidad_sardina
        ]
    })


    st.bar_chart(
        grafico.set_index("Especie")
    )


    st.subheader("📋 Datos utilizados")


    with st.expander(
        "Ver datos utilizados para la predicción"
    ):

        st.dataframe(datos)