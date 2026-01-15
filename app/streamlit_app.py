import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import plotly.graph_objects as go
import plotly.express as px

# Configurar página
st.set_page_config(
    page_title="Data Science App - Consumo de TV",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Agregar directorio raíz al path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from output.models.train import train
from output.models.evaluate import evaluate
from pipelines.consumos_culturales_data_science.etl import run_etl
from output.features.features import run_features_pipeline

# Configuración de rutas
MODEL_PATH = BASE_DIR / "models" / "model_tv.pkl"
TEST_PATH = BASE_DIR / "models" / "test_data.pkl"

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar con opciones
st.sidebar.title("🎛️ Navegación")
app_mode = st.sidebar.radio(
    "Selecciona una opción:",
    ["Home", "Predicción", "Entrenar Modelo", "Métricas del Modelo"]
)

# ============================================================================
# HOME
# ============================================================================
if app_mode == "Home":
    st.title("📺 Predicción de Consumo de Televisión")
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
    
    with col2:
        st.header("Información del Modelo")
        
        if MODEL_PATH.exists():
            st.success("✅ Modelo entrenado disponible")
            model_stats = {
                "Estado": "Listo para usar",
                "Ubicación": str(MODEL_PATH),
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
    
    st.write("Ingresa las características para hacer una predicción:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Opción 1: Entrada Manual")
        st.write("Ingresa los valores manualmente (ajusta según tus features):")
        
        # Crear inputs dinámicos basados en el número de features del modelo
        n_features = model.n_features_in_
        features = []
        
        feature_names = [f"Feature {i+1}" for i in range(n_features)]
        
        for i in range(n_features):
            feature_value = st.number_input(
                f"{feature_names[i]}:",
                value=0.0,
                step=0.1,
                key=f"feature_{i}"
            )
            features.append(feature_value)
        
        if st.button("🔮 Predecir", key="predict_manual"):
            try:
                X = np.array(features).reshape(1, -1)
                prediction = model.predict(X)[0]
                probability = float(model.predict_proba(X).max())
                
                st.success("✅ Predicción realizada")
                
                col_pred1, col_pred2 = st.columns(2)
                with col_pred1:
                    st.metric("Predicción", f"Clase {prediction}", delta=None)
                with col_pred2:
                    st.metric("Confianza", f"{probability*100:.2f}%", delta=None)
                
            except Exception as e:
                st.error(f"Error en la predicción: {str(e)}")
    
    with col2:
        st.subheader("Opción 2: Entrada CSV")
        st.write("Sube un archivo CSV con múltiples predicciones:")
        
        uploaded_file = st.file_uploader("Elige un archivo CSV", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.write("Primeras filas del archivo:")
                st.dataframe(df.head())
                
                if st.button("🔮 Predecir CSV", key="predict_csv"):
                    predictions = model.predict(df)
                    probabilities = model.predict_proba(df).max(axis=1)
                    
                    results_df = df.copy()
                    results_df["Predicción"] = predictions
                    results_df["Confianza"] = probabilities * 100
                    
                    st.success(f"✅ {len(results_df)} predicciones realizadas")
                    st.dataframe(results_df)
                    
                    # Botón para descargar resultados
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar resultados",
                        data=csv,
                        file_name="predicciones.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error procesando CSV: {str(e)}")

# ============================================================================
# ENTRENAR MODELO
# ============================================================================
elif app_mode == "Entrenar Modelo":
    st.title("🤖 Entrenar Nuevo Modelo")
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
        if st.button("🚀 Iniciar Entrenamiento", key="train_btn"):
            with st.spinner("Entrenando modelo..."):
                try:
                    train()
                    st.success("✅ Modelo entrenado exitosamente")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Error en el entrenamiento: {str(e)}")

# ============================================================================
# MÉTRICAS
# ============================================================================
elif app_mode == "Métricas del Modelo":
    st.title("📊 Métricas y Evaluación del Modelo")
    st.markdown("---")
    
    if not TEST_PATH.exists():
        st.error("❌ Datos de prueba no encontrados. Entrena el modelo primero.")
        st.stop()
    
    try:
        # Evaluar modelo
        with st.spinner("Cargando métricas..."):
            metrics = evaluate()
        
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
