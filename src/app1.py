import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Configuración de la página
st.set_page_config(
    page_title="Predicción y Mapa CalCOFI - Especies Marinas",  
    layout="wide"
)

st.title("Explorador y Predictor de peces en California")
st.write("Visualiza las estaciones de muestreo en la costa de California, filtra por fecha, coordenadas y simula las condiciones ambientales.")

# 2. Cargar los modelos

@st.cache_resource
def load_models():
    try:
        # Obtenemos la ruta absoluta del archivo actual (src/app1.py)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Subimos un nivel para llegar a la raíz del proyecto (donde está 'notebooks' y 'src')
        root_dir = os.path.abspath(os.path.join(current_dir, ".."))
        
        # Construimos las rutas absolutas directamente desde la raíz
        path_sardina = os.path.join(root_dir, "notebooks", "individuales", "modelo_sardina_rf.pkl")
        path_anchoa = os.path.join(root_dir, "notebooks", "individuales", "modelo_anchoa_rf.pkl")
        
        modelo_sardina = joblib.load(path_sardina)
        modelo_anchoa = joblib.load(path_anchoa)
        return modelo_sardina, modelo_anchoa
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}")
        return None, None

# 3. Sidebar: Panel de Control Completo
st.sidebar.header("🎛️ Panel de Control")

especie_seleccionada = st.sidebar.selectbox(
    "Seleccionar Especie", 
    options=["Anchoa (Engraulis mordax)", "Sardina (Sardinops sagax)"]
)

st.sidebar.subheader("Filtros Temporales")
selected_year = st.sidebar.slider("Año", min_value=1950, max_value=2020, value=1990, step=1)
selected_month = st.sidebar.slider("Mes", min_value=1, max_value=12, value=6, step=1)

st.sidebar.subheader("📍 Coordenadas Geográficas")
lat = st.sidebar.slider("Latitud", 30.0, 35.0, 32.0, 0.5)
lon = st.sidebar.slider("Longitud", -124.0, -117.0, -118.5, 0.5)

model_activo = modelo_anchoa if especie_seleccionada.startswith("Anchoa") else modelo_sardina

if model_activo is not None:
    expected_features = model_activo.feature_names_in_

    # Layout principal en dos columnas: Izquierda el Mapa dinámico, Derecha los controles físicos y gráficos
    col_mapa, col_controles = st.columns([1.2, 1])

    with col_mapa:
        st.subheader("Ubicación Actual en el Mapa")
        st.write(f"Mostrando estación en Lat: **{lat}**, Lon: **{lon}**")
        
        # DataFrame dinámico con la única coordenada seleccionada por el usuario
        df_punto_actual = pd.DataFrame({
            'lat': [lat],
            'lon': [lon]
        })
        
        # Mapa nativo de Streamlit centrado interactivamente en la coordenada elegida
        st.map(df_punto_actual, latitude='lat', longitude='lon', zoom=6)
        st.info("El punto rojo en el mapa se actualiza en tiempo real según los sliders de la izquierda.")

    with col_controles:
        st.subheader("Condiciones Físico-Químicas")
        
        t_degc = st.slider("Temperatura del agua (°C)", 8.0, 25.0, 15.0, 0.1)
        o2 = st.slider("Oxígeno disuelto (ml/L)", 0.5, 8.0, 5.0, 0.1)

        # Gráfico de barras visual para los valores de Temperatura y Oxígeno
        df_barras = pd.DataFrame({
            'Variable': ['Temperatura (°C)', 'Oxígeno (ml/L)'],
            'Valor': [t_degc, o2]
        }).set_index('Variable')
        
        st.bar_chart(df_barras, height=200)

    # 4. Preparar datos y ejecutar predicción
    input_data = {}
    for col in expected_features:
        col_lower = col.lower()
        if 'lat' in col_lower:
            input_data[col] = lat
        elif 'lon' in col_lower:
            input_data[col] = lon
        elif 'year' in col_lower:
            input_data[col] = selected_year
        elif 'month' in col_lower:
            input_data[col] = selected_month
        elif 't_deg' in col_lower or 'temp' in col_lower:
            input_data[col] = t_degc
        elif 'o2' in col_lower:
            input_data[col] = o2
        else:
            input_data[col] = 0.0

    df_usuario = pd.DataFrame([input_data])[expected_features]

    st.divider()
    if st.button("🚀 Calcular Predicción de Abundancia", type="primary", use_container_width=True):
        pred_log = model_activo.predict(df_usuario)[0]
        pred_real = np.expm1(pred_log)
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric(label=f"Predicción Escala Real ({especie_seleccionada})", value=f"{max(0, pred_real):,.2f} ind.")
        with col_res2:
            st.metric(label="Valor en Log-Espacio", value=f"{pred_log:.4f}")
            
        st.info(f"Simulación para el año **{selected_year}**, mes **{selected_month}** en la coordenada (Lat: {lat}, Lon: {lon}).")
else:
    st.warning("Los modelos no están disponibles.")