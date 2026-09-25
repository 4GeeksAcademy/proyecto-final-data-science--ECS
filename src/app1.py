import os
import joblib
import streamlit as st

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Predicción de Especies Marinas",
    page_icon="🐟",
    layout="wide"
)

# ==========================================
# FUNCIONES DE CARGA SEPARADA CON MMAP_MODE
# (Rutas originales + Ahorro crítico de RAM)
# ==========================================
@st.cache_resource
def load_modelo_sardina():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path_sardina = os.path.join(current_dir, "..", "notebooks", "individuales", "modelo_sardina_rf.pkl")
    # mmap_mode='r' evita que el modelo sature la memoria RAM de Render al arrancar
    return joblib.load(path_sardina, mmap_mode='r')

@st.cache_resource
def load_modelo_anchoa():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path_anchoa = os.path.join(current_dir, "..", "notebooks", "individuales", "modelo_anchoa_rf.pkl")
    return joblib.load(path_anchoa, mmap_mode='r')

# ==========================================
# INTERFAZ PRINCIPAL DE STREAMLIT
# ==========================================
st.title("🌊 Predicción de Especies Marinas")
st.write("Aplicación optimizada para la predicción de Sardina y Anchoa en Render.")

# Pestañas para separar la ejecución de cada modelo de manera independiente
pestana_sardina, pestana_anchoa = st.tabs(["🐟 Predicción Sardina", "🦐 Predicción Anchoa"])

# ==========================================
# SECCIÓN SARDINA
# ==========================================
with pestana_sardina:
    st.header("Modelo de Predicción - Sardina")
    st.write("Introduce los parámetros para realizar la predicción de sardina.")
    
    # Espacio para tus inputs (ejemplo: st.number_input)
    # var_s1 = st.number_input("Variable 1 (Sardina)", value=0.0)
    
    if st.button("Predecir Sardina"):
        try:
            with st.spinner("Cargando modelo de sardina desde disco..."):
                modelo_sardina = load_modelo_sardina()
            
            # Ejemplo de predicción:
            # resultado = modelo_sardina.predict([[var_s1]])
            # st.success(f"Resultado: {resultado}")
            
            st.success("¡Modelo de sardina cargado y listo para operar!")
        except Exception as e:
            st.error(f"Error al cargar el modelo de sardina: {e}")

# ==========================================
# SECCIÓN ANCHOA
# ==========================================
with pestana_anchoa:
    st.header("Modelo de Predicción - Anchoa")
    st.write("Introduce los parámetros para realizar la predicción de anchoa.")
    
    # Espacio para tus inputs de anchoa
    # var_a1 = st.number_input("Variable 1 (Anchoa)", value=0.0)
    
    if st.button("Predecir Anchoa"):
        try:
            with st.spinner("Cargando modelo de anchoa desde disco..."):
                modelo_anchoa = load_modelo_anchoa()
            
            # Ejemplo de predicción:
            # resultado_anchoa = modelo_anchoa.predict([[var_a1]])
            # st.success(f"Resultado: {resultado_anchoa}")
            
            st.success("¡Modelo de anchoa cargado y listo para operar!")
        except Exception as e:
            st.error(f"Error al cargar el modelo de anchoa: {e}")