import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import plotly.graph_objects as go
import time
# Configurar página
st.set_page_config(
    page_title="Data Science App - Consumo de TV y Teatro",
    page_icon="📺🎭", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Agregar directorio raíz al path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from output.models.model_tv.train_tv import train_tv, set_training_params
from output.models.model_tv.evaluate_tv import evaluate_tv
from output.models.model_teatro.train_teatro import train_teatro, set_training_params_teatro
from output.models.model_teatro.evaluate_teatro import evaluate_teatro
from pipelines.consumos_culturales_data_science.etl import run_etl
from output.features.features import run_features_pipeline


# Configuración de rutas
MODEL_PATH = BASE_DIR / "output" / "models_pkl_tv" / "model_tv.pkl"
TEST_PATH = BASE_DIR / "output" / "models_pkl_tv" / "test_data_tv.pkl"

MODEL_PATH_TEATRO = BASE_DIR / "output" / "models_pkl_teatro" / "model_teatro.pkl"
TEST_PATH_TEATRO = BASE_DIR / "output" / "models_pkl_teatro" / "test_data_teatro.pkl"


# Cargar estilos CSS desde el archivo externo
css_file = BASE_DIR / "app" / "styles.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        css_content = f.read()
    st.markdown(f'<style>{css_content} /* {time.time()} */</style>', unsafe_allow_html=True)



# Sidebar con opciones
st.sidebar.title("🎛️ Navegación")
app_mode = st.sidebar.radio(
    "Selecciona una opción:",
    ["Home", "Predicción", "Entrenar Modelo TV", "Entrenar Modelo Teatro", "Métricas del Modelo"]
)




# ============================================================================
# HOME
# ============================================================================
if app_mode == "Home":
    st.title("Predicción de Consumo de Televisión 📺 y Teatro 🎭")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("¿Qué es esta aplicación?")
        st.write("""
        Esta aplicación utiliza **Machine Learning** para predecir si una persona
        consume televisión basándose en sus características sociodemográficas.
        
        **Funcionalidades:**
        - 🎯 Hacer predicciones personalizadas
        - 🤖 Entrenar nuevos modelos
        - 📊 Visualizar métricas de desempeño
        - 📈 Analizar resultados
        """)
    
                #Esto me sirvio para debuggear las rutas, todavia sigo teniendo problemas con eso
                #""" st.write("BASE_DIR:", BASE_DIR)
                #st.write("MODEL_PATH:", MODEL_PATH)
                #st.write("Existe TV:", MODEL_PATH.exists())
                #st.write("MODEL_PATH_TEATRO:", MODEL_PATH_TEATRO)
                #st.write("Existe Teatro:", MODEL_PATH_TEATRO.exists()) """
    with col2:
        st.header("Información del Modelo")
        
        if MODEL_PATH.exists() and MODEL_PATH_TEATRO.exists():
            st.success("✅ Modelo entrenado disponible")
            model_stats = {
                "Estado": "Listo para usar",
                "Tipo": "Random Forest Classifier"
            }
            for key, value in model_stats.items():
                st.write(f"**{key}:** {value}")
        else:
            st.warning("⚠️ No hay modelo entrenado. Entrena uno primero.")
    
    st.markdown("---")
    st.subheader("Pasos siguientes:")
    st.info("""
    1. Dirígete a **Entrenar Modelo** para crear un nuevo modelo
    2. Usa **Predicción** para hacer predicciones individuales
    3. Consulta **Métricas del Modelo** para evaluar el desempeño
    """)

# ============================================================================
# PREDICCIÓN
# ============================================================================
elif app_mode == "Predicción":
    st.title("🎯 Realizar Predicción")
    st.markdown("---")
    
    if not MODEL_PATH.exists():
        st.error("❌ Modelo no encontrado. Por favor, entrena un modelo primero.")
        st.stop()
    
    # Cargar modelo
    model = joblib.load(MODEL_PATH)
    
    st.write("Carga un archivo CSV para hacer predicciones:")
    st.markdown("---")
    
    st.subheader("📤 Carga tu archivo CSV")
    st.write("El archivo debe contener las mismas columnas de entrada que el modelo fue entrenado.")
    
    uploaded_file = st.file_uploader("Elige un archivo CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.write("**Primeras filas del archivo:**")
            st.dataframe(df.head(), use_container_width=True)
            
            st.info(f"📊 Total de registros: {len(df)}")
            
            if st.button("🔮 Realizar Predicciones", key="predict_csv"):
                with st.spinner("Procesando predicciones..."):
                    try:
                        predictions = model.predict(df)
                        probabilities = model.predict_proba(df).max(axis=1)
                        
                        results_df = df.copy()
                        results_df["Predicción"] = predictions
                        results_df["Confianza (%)"] = probabilities * 100
                        
                        st.success(f"✅ {len(results_df)} predicciones realizadas exitosamente")
                        st.markdown("---")
                        
                        st.write("**Resultados de predicción:**")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Estadísticas
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total predicciones", len(results_df))
                        with col2:
                            st.metric("Confianza promedio", f"{probabilities.mean()*100:.2f}%")
                        with col3:
                            confianza_min = f"{probabilities.min()*100:.2f}%"
                            st.metric("Confianza mínima", confianza_min)
                        
                        st.markdown("---")
                        
                        # Botón para descargar resultados
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar resultados",
                            data=csv,
                            file_name="predicciones.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"❌ Error al realizar predicciones: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error al procesar CSV: {str(e)}")

# ============================================================================
# ENTRENAR MODELO TV
# ============================================================================
elif app_mode == "Entrenar Modelo TV":
    st.title("📺 Entrenar Modelo de Televisión")
    st.markdown("---")
    
    st.warning("⚠️ Este proceso puede tardar algunos minutos...")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("""
        Al hacer clic en el botón, se ejecutará el pipeline completo:
        1. **ETL** - Limpieza y transformación de datos
        2. **Features** - Ingeniería de características
        3. **Entrenamiento** - Entrenamiento del modelo
        4. **Guardado** - Almacenamiento del modelo entrenado
        """)
    
    with col2:
        st.subheader("⚙️ Configuración")
        feature_method = st.radio(
            "Método de features:",
            ["OneHotEncoder", "CatPCA"],
            help="Elige el método para procesar variables categóricas"
        )
        
        # Parámetros adicionales para CatPCA
        if feature_method == "CatPCA":
            n_components = st.slider(
                "Número de componentes:",
                min_value=2,
                max_value=8,
                value=5,
                help="Número de componentes principales para CatPCA"
            )
        else:
            n_components = None
        
        if st.button("🚀 Iniciar Entrenamiento TV", key="train_btn_tv"):
            with st.spinner("Entrenando modelo..."):
                try:
                    st.info(f"Usando método: {feature_method}")
                    if feature_method == "CatPCA":
                        st.info(f"Componentes: {n_components}")
                    
                    # Establecer parámetros de entrenamiento
                    method_key = "catpca" if feature_method == "CatPCA" else "onehot"
                    set_training_params(method=method_key, n_components=n_components)
                    
                    train_tv()
                    st.success("✅ Modelo de TV entrenado exitosamente")
                    st.snow()
                except Exception as e:
                    st.error(f"❌ Error en el entrenamiento: {str(e)}")

# ============================================================================
# ENTRENAR MODELO TEATRO
# ============================================================================
elif app_mode == "Entrenar Modelo Teatro":
    st.title("🎭 Entrenar Modelo de Teatro")
    st.markdown("---")
    
    st.warning("⚠️ Este proceso puede tardar algunos minutos...")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("""
        Al hacer clic en el botón, se ejecutará el pipeline completo:
        1. **ETL** - Limpieza y transformación de datos
        2. **Features** - Ingeniería de características
        3. **Entrenamiento** - Entrenamiento del modelo
        4. **Guardado** - Almacenamiento del modelo entrenado
        """)
    
    with col2:
        st.subheader("⚙️ Configuración")
        feature_method = st.radio(
            "Método de features:",
            ["OneHotEncoder", "CatPCA"],
            help="Elige el método para procesar variables categóricas",
            key="teatro_radio"
        )
        
        # Parámetros adicionales para CatPCA
        if feature_method == "CatPCA":
            n_components = st.slider(
                "Número de componentes:",
                min_value=2,
                max_value=8,
                value=5,
                help="Número de componentes principales para CatPCA",
                key="teatro_slider"
            )
        else:
            n_components = None
        
        if st.button("🚀 Iniciar Entrenamiento Teatro", key="train_btn_teatro"):
            with st.spinner("Entrenando modelo..."):
                try:
                    st.info(f"Usando método: {feature_method}")
                    if feature_method == "CatPCA":
                        st.info(f"Componentes: {n_components}")
                    
                    # Establecer parámetros de entrenamiento para teatro
                    method_key = "catpca" if feature_method == "CatPCA" else "onehot"
                    set_training_params_teatro(method=method_key, n_components=n_components)
                    
                    train_teatro()
                    st.success("✅ Modelo de Teatro entrenado exitosamente")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Error en el entrenamiento: {str(e)}")

# ============================================================================
# MÉTRICAS
# ============================================================================
elif app_mode == "Métricas del Modelo":
    st.title("📊 Métricas y Evaluación del Modelo")
    st.markdown("---")
    
    # Verificar qué modelos están disponibles (dinámicamente)
    tv_model = BASE_DIR / "output" / "models_pkl_tv" / "model_tv.pkl"
    tv_test = BASE_DIR / "output" / "models_pkl_tv" / "test_data_tv.pkl"
    teatro_model = BASE_DIR / "output" / "models_pkl_teatro" / "model_teatro.pkl"
    teatro_test = BASE_DIR / "output" / "models_pkl_teatro" / "test_data_teatro.pkl"
    
    tv_available =  tv_model.exists() and tv_test.exists()
    teatro_available = teatro_model.exists() and teatro_test.exists()
    
    if not tv_available and not teatro_available:
        st.warning("⚠️ No hay modelos entrenados. Entrena un modelo primero en 'Entrenar Modelo TV' o 'Entrenar Modelo Teatro'.")
        st.info("💡 Después de entrenar, recarga esta página (F5) para ver los resultados.")
        st.stop()
    
    # Selector de modelo
    available_models = []
    if tv_available:
        available_models.append("📺 Televisión")
    if teatro_available:
        available_models.append("🎭 Teatro")
    
    model_type = st.radio(
        "Selecciona el modelo a evaluar:",
        available_models,
        horizontal=True
    )
    
    st.markdown("---")
    
    try:
        # Evaluar modelo
        with st.spinner("Cargando métricas..."):
            if "Televisión" in model_type:
                metrics = evaluate_tv()
            else:
                metrics = evaluate_teatro()
        
        # Mostrar métricas principales
        st.subheader("📈 Métricas Principales")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1']:.4f}")
        
        st.markdown("---")
        
        # Reporte de clasificación
        st.subheader("📋 Reporte de Clasificación")
        
        if isinstance(metrics['classification_report'], dict):
            report_df = pd.DataFrame(metrics['classification_report']).transpose()
            st.dataframe(report_df, use_container_width=True)
        else:
            st.text(metrics['classification_report'])
        
        st.markdown("---")
        
        # Matriz de confusión
        st.subheader("🎯 Matriz de Confusión")
        
        conf_matrix = metrics['confusion_matrix']
        
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=["No Consume", "Consume"],
            y=["No Consume", "Consume"],
            text=conf_matrix,
            texttemplate="%{text}",
            colorscale="Blues"
        ))
        fig.update_layout(
            title="Matriz de Confusión",
            xaxis_title="Predicción",
            yaxis_title="Actual",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Información adicional
        st.subheader("ℹ️ Interpretación")
        st.info("""
        - **Accuracy**: Porcentaje de predicciones correctas
        - **Precision**: De las que predijo como positivas, cuántas eran correctas
        - **Recall**: De las que eran realmente positivas, cuántas detectó
        - **F1-Score**: Balance entre Precision y Recall
        """)
        
    except Exception as e:
        st.error(f"❌ Error al cargar métricas: {str(e)}")

# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8rem;">
    <p>Data Science App | Predicción de Consumo de Televisión</p>
    <p>Desarrollado con Streamlit y Machine Learning</p>
</div>
""", unsafe_allow_html=True)
