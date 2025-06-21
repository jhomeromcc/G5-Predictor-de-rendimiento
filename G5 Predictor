import streamlit as st
import pandas as pd
import pickle

# Título de la aplicación
st.title("Clasificador de EPECIEN - Árbol de Decisión")

# Cargar modelo
model_choice = st.radio(
    "Selecciona el modelo:",
    ('decision_tree_model.pkl', 'best_decision_tree_model.pkl')
)

@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Entradas del usuario
st.subheader("Ingrese los datos del estudiante:")

sexo = st.selectbox("Sexo", ["M", "F"])
edad = st.number_input("Edad", min_value=15, max_value=60, value=18)
beca = st.selectbox("Tipo de Ingreso", ["REGULAR", "BECAS18"])
modalidad = st.selectbox("Modalidad", ["PRESENCIAL", "VIRTUAL"])
carrera = st.selectbox("Carrera", ["Ingeniería", "Educación", "Administración", "Otra"])
tipo_ebr = st.selectbox("Tipo de EBR", ["Estatal", "Privada"])
ebio = st.number_input("EBIO (Examen Biología)", 0.0, 100.0, step=0.1)
equi = st.number_input("EQUI (Examen Química)", 0.0, 100.0, step=0.1)
efis = st.number_input("EFIS (Examen Física)", 0.0, 100.0, step=0.1)
emat = st.number_input("EMAT (Examen Matemática)", 0.0, 100.0, step=0.1)
ecl = st.number_input("ECL (Examen Comunicación)", 0.0, 100.0, step=0.1)

input_data = pd.DataFrame([{
    'SEXO': sexo,
    'EDAD': edad,
    'BECAS18/REGULAR': beca,
    'MODALIDAD': modalidad,
    'CARRERA': carrera,
    'TIPO_EBR': tipo_ebr,
    'EBIO': ebio,
    'EQUI': equi,
    'EFIS': efis,
    'EMAT': emat,
    'ECL': ecl
}])

# Convertir categorías a dummies si fue entrenado así (esto depende del modelo original)
# input_data = pd.get_dummies(input_data)  # solo si se entrenó así

if st.button("Predecir"):
    model = load_model(model_choice)
    try:
        prediction = model.predict(input_data)[0]
        result = "EPECIEN (Aprueba)" if prediction == 0 else "notEPECIEN (No Aprueba)"
        st.success(f"Resultado de la predicción: {result}")
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
