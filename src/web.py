import streamlit as st
import pickle
import numpy as np
import streamlit as np_st  # Evitar conflictos de nombres

# 1. Configurar la página
st.title("Predicción con Random Forest")
st.write("Introduce los valores para predecir si encontrarás Anchoas o Sardinas.")


# 2. Cargar el modelo entrenado
@st.cache_resource
def load_model():
    with open("modelo_sardina_rf.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()

# 3. Crear formularios de entrada en la interfaz
st.sidebar.header("Características del agua")
salnty = st.sidebar.slider("Salinidad", 1, 2, 3)
year = st.sidebar.slider("Año", 1, 2, 3)


# 4. Realizar la predicción
input_data = np.array(
    [[salnty, year]]
)

if st.button("Predecir clase"):
    prediction = model.predict(input_data)
    target_names = ["Anchoas", "Sardinas"]
    resultado = target_names[prediction[0]]

    st.success(f"El modelo predice que hay muchas: **{resultado}**")
