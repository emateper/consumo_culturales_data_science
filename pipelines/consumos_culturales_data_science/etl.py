"""
Carga de archivo Excel con datos de consumos culturales.

En este Script voy a limpiar, transformar y preparar los datos para el modelado. Todo el analisis extra esta en el Notebook asociado.

"""



#Path, una clase moderna de Python para manejar rutas de archivos y carpetas de forma portable (funciona igual en Windows, Linux, Mac).
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

#ruta del archivo Excel
DATA_PATH = BASE_DIR / "data" / "0_raw" / "base-datos-encc-2022-2023.xlsx"

#print de verificación para ver donde estoy parado y que la ruta exista
"""print("BASE_DIR:", BASE_DIR)
print("DATA_PATH:", DATA_PATH)
print("EXISTE:", DATA_PATH.exists())
"""


#carga del archivo Excel en un DataFrame de pandas
df = pd.read_excel(DATA_PATH, sheet_name='base-datos-encc-2022-2023')

"""
    Selecciona las columnas relevantes del dataset ENCC.

    Parámetros
    datos : pd.DataFrame
        DataFrame original con todas las variables.

    Retorna
    pd.DataFrame
        DataFrame filtrado con las columnas de interés.
"""
def seleccionar_columnas(datos: pd.DataFrame) -> pd.DataFrame:

    columnas = [
        "region", "genero",
        "grupos_edad", "niv_socioe", "soc13.1", "soc14.1",
        "tv1", "tv9", "teatro1", "musica9", 
    ]

    return datos[columnas].copy()




def renombrar_columnas(datos: pd.DataFrame) -> pd.DataFrame:
    """
    Renombra las columnas del DataFrame para mayor claridad.

    Parámetros
    datos : pd.DataFrame
        DataFrame con las columnas originales.

    Retorna
    pd.DataFrame
        DataFrame con las columnas renombradas.
    """
    nuevos_nombres = {
        "region": "Region",
        "genero": "Genero",
        "grupos_edad": "Grupo_Edad",
        "niv_socioe": "Nivel_Socioeconomico",
        "soc13.1": "Estudios_Alcanzados",
        "soc14.1": "Trabajo",
        "tv1": "Consumo_Television",
        "tv9": "Consumo_Plataformas_Digitales",
        "teatro1": "Consumo_Teatro",
        "musica9": "Consumo_Musica",
    }

    return datos.rename(columns=nuevos_nombres)




"""
    Recodifica la columna 'Nivel_Socioeconomico' a clases descriptivas.
    datos : pd.DataFrame
       DataFrame que contiene la columna 'Nivel_Socioeconomico'
    Retorna
    pd.DataFrame
        DataFrame con la columna recodificada.
"""
def recodificar_nivel_socioeconomico(datos: pd.DataFrame) -> pd.DataFrame:

    mapeo = {
        "ABC1": "Clase_alta",
        "C2": "Clase_media_alta",
        "C3": "Clase_media",
        "D1": "Clase_media_baja",
        "D2E": "Clase_baja"
    }

    datos = datos.copy()
    datos["Nivel_Socioeconomico"] = datos["Nivel_Socioeconomico"].map(mapeo)

    return datos



def ordenar_estudios_alcanzados(datos: pd.DataFrame) -> pd.DataFrame:
    """
    Define un orden categórico para la variable Estudios_Alcanzados.
    """
    orden_estudios = [
        "Ns Nc",
        "Sin Estudios",
        "Primarios Incompletos",
        "Primarios Completos",
        "Secundarios Incompletos",
        "Secundarios Completos",
        "Terciarios Incompletos",
        "Terciarios Completos",
        "Universitarios Incompletos",
        "Universitarios Completos",
        "Posgrado",
    ]

    datos = datos.copy()
    datos["Estudios_Alcanzados"] = pd.Categorical(
        datos["Estudios_Alcanzados"],
        categories=orden_estudios,
        ordered=True
    )

    return datos


def ordenar_nivel_socioeconomico(datos: pd.DataFrame) -> pd.DataFrame:
    """
    Define un orden categórico para la variable Nivel_Socioeconomico.
    """
    orden_niv_socioe = [
        "Clase_baja",
        "Clase_media_baja",
        "Clase_media",
        "Clase_media_alta",
        "Clase_alta",
    ]

    datos = datos.copy()
    datos["Nivel_Socioeconomico"] = pd.Categorical(
        datos["Nivel_Socioeconomico"],
        categories=orden_niv_socioe,
        ordered=True
    )

    return datos


#Dejo todos los cambios dentro de df_clean y lo exporto a un nuevo archivo Excel en la carpeta de datos procesados.

TRANSFORMACIONES = [
    seleccionar_columnas,
    renombrar_columnas,
    recodificar_nivel_socioeconomico,
    ordenar_estudios_alcanzados,
    ordenar_nivel_socioeconomico,
]

BASE_DIR = Path(__file__).resolve().parents[2]

OUTPUT_PATH = BASE_DIR / "data" / "1_interim" / "consumos_culturales_clean.xlsx"

def ejecutar_pipeline(df: pd.DataFrame, pasos: list) -> pd.DataFrame:
    for paso in pasos:
        df = paso(df)
    return df


def df_clean_final(
    df: pd.DataFrame,
    pasos: list,
    output_path: Path
) -> pd.DataFrame:
    df_clean = ejecutar_pipeline(df, pasos)
    df_clean.to_excel(output_path, index=False)
    return df_clean


def run_etl() -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de limpieza y devuelve el DataFrame limpio.
    """
    df_clean = df_clean_final(
        df=df,
        pasos=TRANSFORMACIONES,
        output_path=OUTPUT_PATH
    )
    return df_clean
