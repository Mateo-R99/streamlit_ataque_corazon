import streamlit as st
import pandas as pd
import pickle

# Configuración inicial de la página
st.set_page_config(
    page_title="Predicción de Inversión - Videojuegos",
    page_icon="🎮",
    layout="centered"
)

# Cargar el modelo, el scaler y las variables desde el archivo .pkl
@st.cache_resource
def cargar_modelo():
    filename = 'modelo-reg.pkl'
    modelo, min_max_scaler, variables = pickle.load(open(filename, 'rb'))
    return modelo, min_max_scaler, variables

try:
    modelo, min_max_scaler, variables = cargar_modelo()
except FileNotFoundError:
    st.error("No se encontró el archivo 'modelo-reg.pkl'. Asegúrate de haberlo subido al entorno de Colab.")
    st.stop()

# Interfaz de usuario en Streamlit
st.title('Predicción de inversión en una tienda de videojuegos')
st.write("Utiliza los controles a continuación para ingresar los datos del cliente y calcular la predicción.")

# Controles de entrada (Widgets)
Edad = st.slider('Edad', min_value=14, max_value=52, value=20, step=1)
videojuego = st.selectbox(
    'Videojuego', 
    ["'Mass Effect'", "'Battlefield'", "'Fifa'", "'KOA: Reckoning'", "'Crysis'", "'Sim City'", "'Dead Space'", "'F1'"]
)
Plataforma = st.selectbox(
    'Plataforma', 
    ["'Play Station'", "'Xbox'", "PC", "Otros"]
)
Sexo = st.selectbox('Sexo', ['Hombre', 'Mujer'])
Consumidor_habitual = st.selectbox('Consumidor_habitual', ['True', 'False'])

# Botón de predicción
if st.button('Calcular Predicción'):
    # Crear el DataFrame inicial con los datos ingresados
    datos = [[Edad, videojuego, Plataforma, Sexo, Consumidor_habitual]]
    data = pd.DataFrame(datos, columns=['Edad', 'videojuego', 'Plataforma', 'Sexo', 'Consumidor_habitual'])
    
    # 1. Preparación de datos (Variables Dummy)
    data_preparada = data.copy()
    data_preparada = pd.get_dummies(
        data_preparada, 
        columns=['videojuego', 'Plataforma', 'Sexo', 'Consumidor_habitual'], 
        drop_first=False, 
        dtype=int
    )
    
    # 2. Reindexar columnas para que coincidan exactamente con las del entrenamiento
    data_preparada = data_preparada.reindex(columns=variables, fill_value=0)
    
    # 3. Aplicar el modelo para realizar la predicción
    y_pred = modelo.predict(data_preparada)
    
    # Mostrar resultados en la interfaz
    st.subheader("Resultado de la Predicción")
    st.success(f"La inversión predicha es de: **{y_pred[0]:.2f}**")
    st.warning("El modelo tiene un error del 10% (mape: error porcentual)")
