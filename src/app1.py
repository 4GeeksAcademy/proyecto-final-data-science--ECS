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
    path_anchoa = os.path.join(current_dir, "..", "notebooks", "individuales", "modelo_sardina_rf.pkl") 
    return joblib.load(path_anchoa, mmap_mode='r')

# ==========================================
# INTERFAZ PRINCIPAL
# ==========================================
st.title("🌊 Panel de Control: Predicción de Abundancia Marina")
st.write("Sistema optimizado para ejecución estable en Render (California region).")

pestana_sardina, pestana_anchoa = st.tabs(["🐟 Predicción Sardina", "🦐 Predicción Anchoa"])

# ==========================================
# SECCIÓN SARDINA
# ==========================================
with pestana_sardina:
    st.header("Modelo Predictivo - Sardina")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗓️ Filtros Temporales")
        Year_s = st.selectbox("Year", options=list(range(1950, 2027)), index=70, key="year_s")
        Month_s = st.slider("Month", 1, 12, 6, key="month_s")
        
        st.subheader("🗺️ Variables Geográficas")
        lat_round_s = st.slider("lat_round", min_value=32.0, max_value=42.0, value=34.05, step=0.01, key="lat_r_s")
        lon_round_s = st.slider("lon_round", min_value=-124.4, max_value=-114.1, value=-118.24, step=0.01, key="lon_r_s")
        Depthm_s = st.slider("Depthm (Profundidad)", min_value=0.0, max_value=4000.0, value=50.0, step=10.0, key="depth_s")
        
        st.subheader("🧪 Condiciones Fisicoquímicas y Oceanográficas")
        T_degC_s = st.slider("T_degC (Temperatura)", min_value=0.0, max_value=30.0, value=15.0, step=0.1, key="temp_s")
        Salnty_s = st.slider("Salnty (Salinidad)", min_value=30.0, max_value=40.0, value=33.5, step=0.1, key="sal_s")
        O2ml_L_s = st.slider("O2ml_L (Oxígeno)", min_value=0.0, max_value=10.0, value=5.0, step=0.1, key="o2_s")
        STheta_s = st.slider("STheta (Densidad potencial)", min_value=20.0, max_value=30.0, value=25.0, step=0.1, key="sth_s")
        ChlorA_s = st.slider("ChlorA (Clorofila)", min_value=0.0, max_value=50.0, value=1.0, step=0.1, key="chl_s")
        PO4uM_s = st.slider("PO4uM (Fosfato)", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="po4_s")
        
        # Espacio por si encuentras la variable número 12 faltante
        extra_feat_12_s = st.slider("Variable 12 (Pendiente de identificar)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="ext_s")
        
    with col2:
        st.subheader("🗺️ Ubicación en el Mapa (California)")
        df_mapa_s = pd.DataFrame({'lat': [lat_round_s], 'lon': [lon_round_s]})
        st.map(df_mapa_s, zoom=5)

    if st.button("Calcular la abundancia", key="btn_sardina"):
        try:
            with st.spinner("Cargando modelo de sardina y calculando abundancia..."):
                modelo_sardina = load_modelo_sardina()
                
                # Respetando estrictamente el orden de tus columnas anotadas:
                features_sardina = [[
                    Year_s, lon_round_s, lat_round_s, Month_s, Salnty_s, 
                    T_degC_s, O2ml_L_s, STheta_s, Depthm_s, ChlorA_s, PO4uM_s, extra_feat_12_s
                ]]
                
                prediccion_s = modelo_sardina.predict(features_sardina)
                valor_predicho = float(prediccion_s[0])
                
            st.success("¡Cálculo completado con éxito!")
            st.metric(label="📊 Abundancia Predicha (Sardina)", value=f"{valor_predicho:,.2f} individuos / biomasa")
            
        except Exception as e:
            st.error(f"Error al calcular la abundancia: {e}")

# ==========================================
# SECCIÓN ANCHOA
# ==========================================
with pestana_anchoa:
    st.header("Modelo Predictivo - Anchoa")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗓️ Filtros Temporales")
        Year_a = st.selectbox("Year", options=list(range(1950, 2027)), index=70, key="year_a")
        Month_a = st.slider("Month", 1, 12, 6, key="month_a")
        
        st.subheader("🗺️ Variables Geográficas")
        lat_round_a = st.slider("lat_round", min_value=32.0, max_value=42.0, value=34.05, step=0.01, key="lat_r_a")
        lon_round_a = st.slider("lon_round", min_value=-124.4, max_value=-114.1, value=-118.24, step=0.01, key="lon_r_a")
        Depthm_a = st.slider("Depthm (Profundidad)", min_value=0.0, max_value=4000.0, value=50.0, step=10.0, key="depth_a")
        
        st.subheader("🧪 Condiciones Fisicoquímicas y Oceanográficas")
        T_degC_a = st.slider("T_degC (Temperatura)", min_value=0.0, max_value=30.0, value=15.0, step=0.1, key="temp_a")
        Salnty_a = st.slider("Salnty (Salinidad)", min_value=30.0, max_value=40.0, value=33.5, step=0.1, key="sal_a")
        O2ml_L_a = st.slider("O2ml_L (Oxígeno)", min_value=0.0, max_value=10.0, value=5.0, step=0.1, key="o2_a")
        STheta_a = st.slider("STheta (Densidad potencial)", min_value=20.0, max_value=30.0, value=25.0, step=0.1, key="sth_a")
        ChlorA_a = st.slider("ChlorA (Clorofila)", min_value=0.0, max_value=50.0, value=1.0, step=0.1, key="chl_a")
        PO4uM_a = st.slider("PO4uM (Fosfato)", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="po4_a")
        NO3uM_a = st.slider("NO3uM (Nutrientes)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key="ext_a")
        
    with col2:
        st.subheader("🗺️ Ubicación en el Mapa (California)")
        df_mapa_a = pd.DataFrame({'lat': [lat_round_a], 'lon': [lon_round_a]})
        st.map(df_mapa_a, zoom=5)

    if st.button("Calcular la abundancia", key="btn_anchoa"):
        try:
            with st.spinner("Cargando modelo de anchoa y calculando abundancia..."):
                modelo_anchoa = load_modelo_anchoa()
                
                features_anchoa = [[
                    Year_a, lon_round_a, lat_round_a, Month_a, Salnty_a, 
                    T_degC_a, O2ml_L_a, STheta_a, Depthm_a, ChlorA_a, PO4uM_a, extra_feat_12_a
                ]]
                
                prediccion_a = modelo_anchoa.predict(features_anchoa)
                valor_predicho_a = float(prediccion_a[0])
                
            st.success("¡Cálculo completado con éxito!")
            st.metric(label="📊 Abundancia Predicha (Anchoa)", value=f"{valor_predicho_a:,.2f} individuos / biomasa")
            
        except Exception as e:
            st.error(f"Error al calcular la abundancia: {e}")