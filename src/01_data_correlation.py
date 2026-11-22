
# ============================================================
# 01_data_correlation.py
# Integrante 1: Datos, correlación y limpieza
# Proyecto: Agrupamiento de Señales EOG del Lenguaje Blink-to-Speak
# ============================================================

# 1. IMPORTAR LIBRERIAS

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler


# 2. CONFIGURAR RUTAS

# Obtener la carpeta principal del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Ruta del archivo original
RUTA_DATASET = BASE_DIR / "data" / "DataSet_completo(1).xlsx"

# Carpetas donde se guardaran los resultados
CARPETA_FIGURAS = BASE_DIR / "fig"
CARPETA_REPORTE = BASE_DIR / "reporte"

# Crear las carpetas si no existen
CARPETA_FIGURAS.mkdir(exist_ok=True)
CARPETA_REPORTE.mkdir(exist_ok=True)


# 3. CARGAR EL DATASET

print("\n========== CARGA DEL DATASET ==========")

df = pd.read_excel(RUTA_DATASET)

print("Dimensiones originales:", df.shape)
print("\nPrimeras filas:")
print(df.head())


# 4. EXPLORACION Y LIMPIEZA

print("\n========== EXPLORACION DE DATOS ==========")

# Revisar valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

print("\nTotal de valores nulos:", df.isnull().sum().sum())

# Revisar registros duplicados
print("\nRegistros duplicados:", df.duplicated().sum())

# Revisar distribución de las clases
distribucion = df["Clase"].value_counts().sort_index()

print("\nDistribucion de clases:")
print(distribucion)

# Calcular porcentajes por clase
porcentajes = df["Clase"].value_counts(
    normalize=True
).sort_index() * 100

print("\nPorcentaje por clase:")
print(porcentajes)


# 5. GRAFICAR DISTRIBUCION DE CLASES

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="Clase",
    hue="Clase",
    palette="viridis",
    legend=False
)

plt.title("Distribucion de muestras por clase")
plt.xlabel("Clase")
plt.ylabel("Numero de muestras")
plt.tight_layout()

plt.savefig(
    CARPETA_FIGURAS / "distribucion_clases.png",
    dpi=300
)

plt.close()


# 6. ELIMINAR LA COLUMNA LONGITUD

print("\n========== PREPARACION DE CARACTERISTICAS ==========")

# Longitud representa la duracion temporal de la señal.
# Se elimina para evitar que la red dependa de esa variable.

df = df.drop(columns=["Longitud"])

# Separar las variables predictoras (X) y la clase objetivo (y)

X = df.drop(columns=["Clase"])
y = df["Clase"]

print("\nDimensiones de X:", X.shape)
print("Dimensiones de y:", y.shape)


# 7. ESTANDARIZACION

# StandardScaler transforma las caracteristicas para que
# tengan media cercana a 0 y desviacion estandar cercana a 1.

escalador = StandardScaler()

X_escalado = pd.DataFrame(
    escalador.fit_transform(X),
    columns=X.columns,
    index=X.index
)

print("\nEstandarizacion realizada correctamente.")


# 8. MATRIZ DE CORRELACION

print("\n========== ANALISIS DE CORRELACION ==========")

# Calcular correlacion de Pearson entre las caracteristicas

matriz_correlacion = X_escalado.corr()

# Graficar mapa de calor

plt.figure(figsize=(16, 12))

sns.heatmap(
    matriz_correlacion,
    cmap="coolwarm",
    center=0,
    annot=False
)

plt.title("Matriz de correlacion de las caracteristicas")
plt.tight_layout()

plt.savefig(
    CARPETA_FIGURAS / "matriz_correlacion.png",
    dpi=300
)

plt.close()


# 9. IDENTIFICAR CARACTERISTICAS ALTAMENTE CORRELACIONADAS

UMBRAL = 0.80

# Tomar valores absolutos para detectar correlaciones
# positivas y negativas.

correlacion_absoluta = matriz_correlacion.abs()

# Conservar solamente la mitad superior de la matriz
# para evitar analizar dos veces el mismo par.

triangulo_superior = correlacion_absoluta.where(
    np.triu(
        np.ones(correlacion_absoluta.shape),
        k=1
    ).astype(bool)
)

# Obtener todos los pares cuya correlacion supera 0.80

pares_correlacionados = []

for columna in triangulo_superior.columns:

    for indice in triangulo_superior.index:

        valor = triangulo_superior.loc[indice, columna]

        if pd.notna(valor) and valor > UMBRAL:

            pares_correlacionados.append(
                (indice, columna, valor)
            )

print("\nPares con correlacion absoluta mayor a", UMBRAL)

for variable1, variable2, valor in pares_correlacionados:

    print(
        f"{variable1} <-> {variable2}: {valor:.4f}"
    )


# 10. ELIMINAR CARACTERISTICAS REDUNDANTES

# Estrategia:
# Se conserva la primera variable y se elimina una variable
# posterior cuando esta tiene correlacion mayor a 0.80
# con alguna de las variables ya conservadas.

caracteristicas_conservar = []
caracteristicas_eliminar = []

for columna in X.columns:

    redundante = False

    for conservada in caracteristicas_conservar:

        correlacion = abs(
            matriz_correlacion.loc[conservada, columna]
        )

        if correlacion > UMBRAL:

            redundante = True
            break

    if redundante:

        caracteristicas_eliminar.append(columna)

    else:

        caracteristicas_conservar.append(columna)


print("\nCaracteristicas eliminadas:")
print(caracteristicas_eliminar)

print("\nCaracteristicas conservadas:")
print(caracteristicas_conservar)


# 11. GENERAR DATASET REDUCIDO

# Se conservan los valores originales de las caracteristicas,
# no los estandarizados, para que el siguiente integrante
# pueda ajustar StandardScaler dentro de cada particion
# de entrenamiento durante la validacion cruzada.

X_reducido = X[caracteristicas_conservar]

# Agregar nuevamente la clase objetivo

dataset_reducido = X_reducido.copy()
dataset_reducido["Clase"] = y

# Exportar archivo Excel

RUTA_SALIDA = BASE_DIR / "Dataset_reducido.xlsx"

dataset_reducido.to_excel(
    RUTA_SALIDA,
    index=False
)


# 12. RESUMEN FINAL

print("\n========== RESULTADOS FINALES ==========")

print("Registros originales:", len(df))
print("Caracteristicas iniciales:", X.shape[1])
print("Caracteristicas eliminadas:", len(caracteristicas_eliminar))
print("Caracteristicas finales:", X_reducido.shape[1])

print("\nDimensiones del dataset reducido:")
print(dataset_reducido.shape)

print("\nArchivo generado:", RUTA_SALIDA)

print("\nGraficas guardadas en:", CARPETA_FIGURAS)

print("\nPROCESO FINALIZADO CORRECTAMENTE.")
