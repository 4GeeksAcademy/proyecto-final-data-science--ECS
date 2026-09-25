import os
import joblib
import pandas as pd
import streamlit as st

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Predicción de Abundancia - Especies Marinas",
    page_icon="🌊",
    layout="wide"
)

# ==========================================
# FUNCIONES DE CARGA LOCAL CON MMAP_MODE (AHORRO DE RAM)
# ==========================================
@st.cache_resource
def load_modelo_sardina():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path_sardina = os.path.join(current_dir, "..", "notebooks", "individuales", "modelo_sardina_rf.pkl")
    return joblib.load(path_sardina, mmap_mode='r')

@st.cache_resource
def load_modelo_anchoa():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path_anchoa = os.path.join(current_dir, "..", "notebooks", "individuales", "modelo_anchoa_rf.pkl") 
    return joblib.load(path_anchoa, mmap_mode='r')

# ==========================================
# INTERFAZ PRINCIPAL
# ==========================================
st.title("🌊 Panel de Control: Predicción de Abundancia Marina")
st.write("Sistema optimizado para ejecución estable en Render (California region).")

# Layout principal dividido en dos columnas: Controles a la izquierda, Mapa y Botón a la derecha
col_ctrl, col_map = st.columns([1, 1], gap="large")

with col_ctrl:
    st.subheader("🎛️ Panel de Control")
    
    # Selector de especie limpio
    especie_seleccionada = st.radio(
        "Seleccionar Especie a Predecir:", 
        ["Sardina", "Anchoa"], 
        horizontal=True
    )
    
    st.markdown("---")
    
    # Filtros Temporales (Sliders)
    st.subheader("🗓️ Filtros Temporales")
    Year = st.slider("Año", min_value=1950, max_value=2026, value=2000, step=1)
    Month = st.slider("Mes", min_value=1, max_value=12, value=6, step=1)
    
    # Coordenadas geográficas y profundidad (Sliders)
    st.subheader("🗺️ Ubicación y Geografía")
    lat_round = st.slider("Latitud", min_value=32.0, max_value=42.0, value=34.05, step=0.01)
    lon_round = st.slider("Longitud", min_value=-124.4, max_value=-114.1, value=-118.24, step=0.01)
    Depthm = st.slider("Profundidad", min_value=0.0, max_value=4000.0, value=50.0, step=10.0)
    
    # Condiciones Físico-Químicas principales (Sliders)
    st.subheader("🧪 Condiciones Físico-Químicas")
    T_degC = st.slider("Temperatura del agua (°C)", min_value=0.0, max_value=30.0, value=15.0, step=0.1)
    PO4uM = st.slider("Nutrientes (Fosfato - PO4uM)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

    # Variables secundarias ocultas para satisfacer las 12 características exactas del modelo
    with st.expander("⚙️ Avanzado (Opcional)"):
        Salnty = st.slider("Salinidad (Salnty)", 30.0, 40.0, 33.5, step=0.1)
        O2ml_L = st.slider("Oxígeno disuelto (O2ml_L)", 0.0, 10.0, 5.0, step=0.1)
        STheta = st.slider("Densidad potencial (STheta)", 20.0, 30.0, 25.0, step=0.1)
        ChlorA = st.slider("Clorofila (ChlorA)", 0.0, 50.0, 1.0, step=0.1)
        NO3uM = st.slider("Nitrato (NO3uM)", 0.0, 50.0, 5.0, step=0.1)

with col_map:
    st.subheader("🗺️ Mapa de California")
    df_mapa = pd.DataFrame({'lat': [lat_round], 'lon': [lon_round]})
    st.map(df_mapa, zoom=5)
    
    st.markdown("---")
    
    # Botón principal de cálculo
    if st.button("Calcular la abundancia de peces", use_container_width=True):
        try:
            with st.spinner(f"Cargando modelo de {especie_seleccionada} y calculando..."):
                if especie_seleccionada == "Sardina":
                    modelo = load_modelo_sardina()
                else:
                    modelo = load_modelo_anchoa()
                
                # Orden exacto de las 12 características requeridas por el modelo
                features = [[
                    Year, lon_round, lat_round, Month, Salnty, 
                    T_degC, O2ml_L, STheta, Depthm, ChlorA, PO4uM, NO3uM
                ]]
                
                prediccion = modelo.predict(features)
                valor_predicho = float(prediccion[0])
                
            st.success("¡Cálculo completado con éxito!")
            
            # Resultado destacado en métrica
            st.metric(
                label=f"📊 Abundancia Predicha ({especie_seleccionada})", 
                value=f"{valor_predicho:,.2f} individuos"
            )
            
            st.info(f"Parámetros: Año {Year}, Mes {Month} | Profundidad: {Depthm}m | Temp: {T_degC}°C | Ubicación: ({lat_round}, {lon_round})")
            
        except Exception as e:
            st.error(f"Error al calcular la abundancia: {e}")