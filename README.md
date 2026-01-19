# 🎭📺 Consumos Culturales en Argentina — Data Science Project

Aplicación de **Data Science end‑to‑end** para analizar y predecir el consumo de **televisión** y **teatro** en Argentina utilizando datos reales de la ENCC (2022–2023).

Incluye:

* Pipeline ETL
* Feature Engineering
* Entrenamiento de modelos ML
* Evaluación
* Aplicación interactiva en **Streamlit**

---

## 🚀 Demo

La aplicación permite:

* Entrenar modelos para TV y Teatro
* Comparar métodos de features (OneHot / CatPCA)
* Visualizar métricas
* Realizar predicciones

---

## 🧠 Arquitectura del proyecto

```
Proyecto Data Science
│
├── app/                      # Streamlit App
│   ├── streamlit_app.py
│   └── styles.css
│
├── data/
│   ├── 0_raw/                # Datos originales
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
├── models_pkl_teatro/
│
├── notebooks/                # Exploración
├── requirements.txt
└── README.md
```

---

## ⚙️ Tecnologías utilizadas

| Área          | Herramientas                        |
| ------------- | ----------------------------------- |
| Lenguaje      | Python 3.10+                        |
| Data          | Pandas, NumPy                       |
| ML            | Scikit‑learn                        |
| Visualización | Streamlit                           |
| Pipelines     | sklearn Pipeline, ColumnTransformer |
| Versionado    | Git + GitHub                        |

---

## 🔄 Flujo del sistema

1. **ETL**

   * Selección de variables
   * Renombrado
   * Recodificación
   * Variables categóricas ordenadas

2. **Features**

   * OneHot Encoding
   * Standard Scaling
   * (Opcional) CatPCA

3. **Modelos**

   * RandomForestClassifier
   * Modelos separados para TV y Teatro

4. **App Streamlit**

   * Interfaz de entrenamiento
   * Evaluación
   * Predicciones

---

## 🧪 Cómo ejecutar el proyecto

### 1️⃣ Clonar repositorio

```bash
git clone <repo-url>
cd proyecto-data-science
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

### 4️⃣ Ejecutar aplicación

```bash
streamlit run app/streamlit_app.py
```

---

## 🤖 Entrenamiento de modelos

Desde la app o directamente:

```bash
python output/models/model_tv/train_tv.py
python output/models/model_teatro/train_teatro.py
```

Los modelos se guardan automáticamente en:

```
models_pkl_tv/
models_pkl_teatro/
```

*(Estas carpetas no se versionan)*

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

## 🧩 Posibles mejoras futuras

* MLflow para tracking de experimentos
* Validación cruzada
* XGBoost / LightGBM
* Feature importance
* Dockerización
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

🎯 *Proyecto diseñado con estructura profesional orientada a entornos reales de Data Science & MLOps.*
