import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Predicción de Ataque al Corazón / Stroke",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Despliegue de Modelo Predictivo: Ataque al Corazón / Stroke")
st.markdown("""
Esta aplicación permite cargar un dataset, entrenar y evaluar modelos de clasificación mediante **Validación Cruzada Estratificada** (`StratifiedKFold`), y realizar predicciones interactivas para nuevos pacientes.
""")

# --- SECCIÓN 1: CARGA DE DATOS ---
st.sidebar.header("📁 1. Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu dataset (Excel o CSV)", type=["xlsx", "csv"])

@st.cache_data
def load_default_data():
    try:
        return pd.read_excel("ataque_corazon.xlsx", sheet_name=0)
    except:
        return None

if uploaded_file is not None:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    st.sidebar.success("¡Dataset personalizado cargado con éxito!")
else:
    df = load_default_data()
    if df is not None:
        st.sidebar.info("Usando el dataset por defecto (`ataque_corazon.xlsx`).")
    else:
        st.warning("Por favor, sube un archivo de datos para comenzar.")
        st.stop()

# Mostrar un vistazo de los datos
with st.expander("🔍 Vista previa del Dataset"):
    st.dataframe(df.head())
    st.write(f"Dimensiones del dataset: {df.shape[0]} filas y {df.shape[1]} columnas.")

# --- SECCIÓN 2: PREPROCESAMIENTO ---
# Variables objetivo y predictoras estándar basadas en tu análisis
objetivo = 'stroke_ataque_corazon'

if objetivo not in df.columns:
    st.error(f"El dataset no contiene la columna objetivo requerida: `{objetivo}`")
    st.stop()

# Limpieza y conversión de tipos
for col in ['hypertension', 'heart_disease', 'ever_married', 'smoking_status', objetivo]:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Separar X e Y
X_raw = df.drop(objetivo, axis=1)
Y_raw = df[objetivo]

# Codificar la variable objetivo si es categórica
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y_raw)

# Codificación Dummy para variables categóricas
cat_cols = X_raw.select_dtypes(include=['category', 'object']).columns.tolist()
num_cols = X_raw.select_dtypes(include=['number']).columns.tolist()

X = pd.get_dummies(X_raw, columns=cat_cols, drop_first=False, dtype=int)

# Normalización de variables numéricas
scaler = MinMaxScaler()
if len(num_cols) > 0:
    X[num_cols] = scaler.fit_transform(X[num_cols])

# --- SECCIÓN 3: ENTRENAMIENTO Y VALIDACIÓN CRUZADA ---
st.header("⚙️ 2. Validación Cruzada de Modelos")

col1, col2 = st.columns(2)
with col1:
    n_splits = st.slider("Número de divisiones (K-Folds)", min_value=3, max_value=10, value=10)
with col2:
    model_choice = st.selectbox(
        "Selecciona el algoritmo principal:",
        ["K-Nearest Neighbors (KNN)", "Random Forest", "Árbol de Decisión (Decision Tree)"]
    )

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
scoring = ('f1_macro', 'accuracy', 'precision_macro', 'recall_macro')

if model_choice == "K-Nearest Neighbors (KNN)":
    k_neighbors = st.slider("Número de vecinos (K)", min_value=1, max_value=15, value=3)
    model = KNeighborsClassifier(n_neighbors=k_neighbors, metric='euclidean')
elif model_choice == "Random Forest":
    n_est = st.slider("Número de estimadores (Árboles)", min_value=50, max_value=300, value=150, step=50)
    model = RandomForestClassifier(n_estimators=n_est, max_depth=15, min_samples_leaf=10, random_state=42)
else:
    max_d = st.slider("Profundidad máxima del árbol", min_value=3, max_value=50, value=15)
    model = DecisionTreeClassifier(max_depth=max_d, min_samples_leaf=10, random_state=42)

if st.button("🚀 Ejecutar Validación Cruzada"):
    with st.spinner("Entrenando y evaluando el modelo..."):
        scores = cross_validate(model, X, Y, cv=cv, scoring=scoring, return_train_score=True)
        scores_df = pd.DataFrame(scores)
        
        st.success("¡Validación cruzada completada!")
        
        # Métricas promedio
        mean_scores = scores_df.mean()
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Accuracy (Test)", f"{mean_scores['test_accuracy']:.4f}")
        col_m2.metric("F1-Score Macro (Test)", f"{mean_scores['test_f1_macro']:.4f}")
        col_m3.metric("Precision Macro (Test)", f"{mean_scores['test_precision_macro']:.4f}")
        col_m4.metric("Recall Macro (Test)", f"{mean_scores['test_recall_macro']:.4f}")
        
        with st.expander("📊 Ver detalles completos por iteración (Folds)"):
            st.dataframe(scores_df)

# --- SECCIÓN 4: PREDICCIÓN EN VIVO PARA NUEVOS PACIENTES ---
st.header("🩺 3. Predicción Interactiva para un Nuevo Paciente")

# Entrenar el modelo con todo el dataset disponible para las predicciones en vivo
model.fit(X, Y)

with st.form("prediction_form"):
    st.subheader("Ingrese los datos clínicos del paciente:")
    f_col1, f_col2 = st.columns(2)
    
    input_data = {}
    with f_col1:
        if 'age' in X_raw.columns:
            input_data['age'] = st.number_input("Edad (años)", min_value=1, max_value=120, value=45)
        if 'avg_glucose_level' in X_raw.columns:
            input_data['avg_glucose_level'] = st.number_input("Nivel Promedio de Glucosa", min_value=40.0, max_value=300.0, value=100.0)
        if 'hypertension' in X_raw.columns:
            input_data['hypertension'] = st.selectbox("Hipertensión", ["No", "Yes"])
        if 'heart_disease' in X_raw.columns:
            input_data['heart_disease'] = st.selectbox("Enfermedad Cardíaca Previa", ["No", "Yes"])

    with f_col2:
        if 'ever_married' in X_raw.columns:
            input_data['ever_married'] = st.selectbox("Casado/a alguna vez", ["No", "Yes"])
        if 'smoking_status' in X_raw.columns:
            input_data['smoking_status'] = st.selectbox("Estado de Tabaquismo", ['Unknown', "'never smoked'", "'formerly smoked'", 'smokes'])

    submit_button = st.form_submit_button(label="🔮 Predecir Riesgo")

if submit_button:
    # Construir dataframe con la entrada del usuario
    input_df = pd.DataFrame([input_data])
    
    # Asegurar tipos y procesar igual que el dataset de entrenamiento
    for col in cat_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype('category')
            
    input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=False, dtype=int)
    
    # Alinear columnas con X para mantener exactamente el mismo espacio de características (features)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
    
    # Normalizar numéricas
    if len(num_cols) > 0:
        input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])
        
    # Realizar predicción
    prediction = model.predict(input_encoded)[0]
    prediction_proba = model.predict_proba(input_encoded)[0]
    
    class_name = label_encoder.inverse_transform([prediction])[0]
    
    st.markdown("---")
    st.subheader("Resultado del Diagnóstico:")
    if str(class_name).lower() in ['yes', '1', 'si']:
        st.error(f"⚠️ **Resultado:** Alto riesgo detectado ({class_name}).")
    else:
        st.success(f"✅ **Resultado:** Bajo riesgo / Sin indicios de ataque al corazón detectados ({class_name}).")
    
    st.write(f"Probabilidades estimadas: **No:** {prediction_proba[0]*100:.2f}% | **Sí:** {prediction_proba[1]*100:.2f}%")