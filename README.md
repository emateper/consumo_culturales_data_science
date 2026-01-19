# 🎭📺 Consumos Culturales en Argentina — Data Science Project

Proyecto **end‑to‑end de Data Science & ML Engineering** para analizar y predecir el consumo de **televisión** y **teatro** en Argentina utilizando datos reales de la ENCC (2022–2023).

Incluye:

* Pipeline ETL reproducible
* Feature Engineering avanzado (One‑Hot / CatPCA)
* Entrenamiento y evaluación de modelos ML
* Serving mediante **API REST (FastAPI)**
* Aplicación interactiva en **Streamlit**

---

## 🚀 Demo

La aplicación permite:

* Entrenar modelos para TV y Teatro
* Comparar métodos de features (OneHot vs CatPCA)
* Visualizar métricas de desempeño
* Realizar predicciones interactivas
* Consumir el modelo vía API

---

## 🧠 Arquitectura del proyecto

```
Proyecto Data Science
│
├── app/                      # Streamlit App
│   ├── streamlit_app.py
│   └── styles.css
│
├── serve/                    # API REST con FastAPI
│   └── app.py
│
├── data/
│   ├── 0_raw/                # Datos originales (ignorado por git)
│   └── 1_interim/            # Datos procesados (ignorado por git)
│
├── pipelines/
│   └── consumos_culturales_data_science/
│       └── etl.py            # Pipeline de limpieza
│
├── output/
│   ├── features/             # Feature engineering
│   └── models/
│       ├── model_tv/
│       └── model_teatro/
│
├── models_pkl_tv/            # Modelos entrenados (ignorado)
├── models_pkl_teatro/        # Modelos entrenados (ignorado)
│
├── notebooks/                # Exploración
├── requirements.txt
└── README.md
```

---

## ⚙️ Tecnologías utilizadas

### 🐍 Lenguaje y entorno

* Python 3.10+
* Entornos virtuales (venv)

### 📊 Procesamiento de datos

* Pandas
* NumPy

### 🤖 Machine Learning

* Scikit‑learn

  * RandomForestClassifier
  * Pipeline
  * ColumnTransformer
  * OneHotEncoder
  * StandardScaler
  * CatPCA (opcional)

### 🧱 Arquitectura y pipelines

* Diseño modular por capas (ETL / Features / Models / Serving)
* Patrón Pipeline
* Tipado con Pydantic

### 🌐 Serving & APIs

* FastAPI
* Uvicorn
* OpenAPI / Swagger UI

### 🖥️ Frontend analítico

* Streamlit

### 🧪 Experimentación

* Jupyter Notebook

### 🗂️ Ingeniería de software

* Git & GitHub
* Estructura profesional de proyecto
* .gitignore para artefactos

---

## 🔄 Flujo del sistema

1. **ETL**

   * Selección de variables
   * Renombrado semántico
   * Recodificación socioeconómica
   * Variables categóricas ordenadas

2. **Feature Engineering**

   * One‑Hot Encoding
   * Escalado estándar
   * Reducción dimensional con CatPCA (opcional)

3. **Modelado**

   * Random Forest por dominio (TV / Teatro)
   * Split estratificado
   * Métricas automáticas

4. **Serving**

   * API REST con endpoints de predicción, entrenamiento y métricas

5. **Visualización**

   * Dashboard interactivo en Streamlit

---

## 🧪 Cómo ejecutar el proyecto

### 1️⃣ Clonar repositorio

```bash
git clone <repo-url>
cd consumo_culturales_data_science
```

### 2️⃣ Crear entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
.venv\Scripts\activate      # Windows
```

### 3️⃣ Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## ▶️ Ejecutar Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## 🌐 Ejecutar API con FastAPI

```bash
uvicorn serve.app:app --reload
```

Abrir documentación interactiva:

```
http://127.0.0.1:8000/docs
```

---

## 🤖 Entrenamiento de modelos

Desde consola:

```bash
python output/models/model_tv/train_tv.py
python output/models/model_teatro/train_teatro.py
```

Los modelos se guardan automáticamente en:

```
models_pkl_tv/
models_pkl_teatro/
```

*(estas carpetas no se versionan)*

---

## 🌐 API — Endpoints principales

| Método | Endpoint | Descripción       |
| ------ | -------- | ----------------- |
| GET    | /        | Info general      |
| GET    | /health  | Estado del modelo |
| POST   | /predict | Predicción        |
| POST   | /train   | Entrenar modelo   |
| GET    | /metrics | Métricas          |

### Ejemplo de request

```json
{
  "features": [1.0, 0.0, 3.0, 2.0]
}
```

### Ejemplo de response

```json
{
  "prediction": 1,
  "probability": 0.87
}
```

---

## 📊 Dataset

Fuente: **Encuesta Nacional de Consumos Culturales (ENCC) 2022–2023**

Variables utilizadas:

* Región
* Género
* Grupo etario
* Nivel socioeconómico
* Nivel educativo
* Situación laboral
* Consumo de TV
* Consumo de plataformas digitales
* Consumo de teatro
* Consumo de música

---

## 🧩 Roadmap / Mejoras futuras

* MLflow para tracking de experimentos
* Validación cruzada
* XGBoost / LightGBM
* Feature importance
* Dockerización
* CI/CD
* Despliegue en la nube

---

## 👨‍💻 Autor

**Emanuel Teper**

* Estudiante de Ciencia de Datos
* Data Scientist Jr
* Interesado en MLOps

---

## ⭐ Si te gustó el proyecto

¡No olvides dejar una estrella ⭐ en el repositorio!

---

## 📝 Licencia

Proyecto con fines educativos y demostrativos.

---

🎯 *Proyecto diseñado con estructura profesional orientada a entornos reales de Data Science, Machine Learning Engineering y MLOps.*
