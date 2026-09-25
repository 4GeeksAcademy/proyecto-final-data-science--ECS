import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Configuración de la página
st.set_page_config(
    page_title="Predicción de Abundancia - CalCOFI", 
    layout="wide"
)

st.title("Predicción de Abundancia de Peces (CalCOFI)")
st.write("Utiliza los controles para simular condiciones ambientales y predecir la abundancia.")

# 2. Cargar los modelos
@st.cache_resource
def load_models():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path_sardina = os.path.join(current_dir, '../notebooks/individuales/modelo_sardina_rf.pkl')
        path_anchoa = os.path.join(current_dir, '../notebooks/individuales/modelo_anchoa_rf.pkl')
        
        modelo_sardina = joblib.load(path_sardina)
        modelo_anchoa = joblib.load(path_anchoa)
        return modelo_sardina, modelo_anchoa
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}")
        return None, None

modelo_sardina, modelo_anchoa = load_models()

# 3. Formulario en la barra lateral
st.sidebar.header("Parámetros Ambientales")
especie_seleccionada = st.sidebar.selectbox(
    "Seleccionar Especie", 
    options=["Anchoa (Engraulis mordax)", "Sardina (Sardinops sagax)"]
)

# Seleccionar el modelo activo para inspeccionar sus columnas exactas
model_activo = modelo_anchoa if especie_seleccionada.startswith("Anchoa") else modelo_sardina

if model_activo is not None:
    # Obtenemos exactamente las columnas que el modelo espera
    expected_features = model_activo.feature_names_in_
    
    st.sidebar.subheader("Valores principales")
    # Sliders para las variables principales que el usuario quiera controlar
    lat = st.sidebar.slider("Latitud", 30.0, 35.0, 32.0, 0.5)
    lon = st.sidebar.slider("Longitud", -124.0, -117.0, -120.0, 0.5)
    year = st.sidebar.number_input("Año", 1950, 2020, 1990)
    month = st.sidebar.slider("Mes", 1, 12, 6)
    t_degc = st.sidebar.slider("Temperatura (°C)", 8.0, 25.0, 15.0, 0.1)
    o2 = st.sidebar.slider("Oxígeno disuelto (ml/L)", 0.5, 8.0, 5.0, 0.1)

    # Construir un diccionario base rellenando todas las columnas que el modelo espera
    # Si alguna columna específica del modelo no está en los controles manuales, se le asigna un valor por defecto (ej. 0.0 o mediana)
    input_data = {}
    for col in expected_features:
        col_lower = col.lower()
        if 'lat' in col_lower:
            input_data[col] = lat
        elif 'lon' in col_lower:
            input_data[col] = lon
        elif 'year' in col_lower:
            input_data[col] = year
        elif 'month' in col_lower:
            input_data[col] = month
        elif 't_deg' in col_lower or 'temp' in col_lower:
            input_data[col] = t_degc
        elif 'o2' in col_lower:
            input_data[col] = o2
        else:
            # Para otras variables que el modelo exija (como ChlorA, Depthm, etc.), ponemos un valor por defecto seguro
            input_data[col] = 0.0

    df_usuario = pd.DataFrame([input_data])
    # Asegurar el orden exacto de columnas que exige el modelo
    df_usuario = df_usuario[expected_features]

    st.subheader("Datos que se enviarán al modelo")
    st.dataframe(df_usuario, use_container_width=True)

    if st.button("🚀 Calcular Abundancia Predicha", type="primary"):
        pred_log = model_activo.predict(df_usuario)[0]
        pred_real = np.expm1(pred_log)
        
        st.divider()
        st.subheader(f"📊 Resultado para: {especie_seleccionada}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Predicción en Escala Real (Individuos)", value=f"{max(0, pred_real):,.2f}")
        with col2:
            st.metric(label="Predicción en Log-Espacio", value=f"{pred_log:.4f}")
else:
    st.warning("Los modelos no están disponibles.")