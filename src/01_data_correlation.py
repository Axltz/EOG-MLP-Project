# ============================================================
# 01_data_correlation.py
# Integrante 1: Preprocesamiento, Z-score, Correlación y Limpieza
# Proyecto: Agrupamiento de Señales EOG del Lenguaje Blink-to-Speak
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importar configuración global
from config import (
    RUTA_DATASET_COMPLETO,
    RUTA_DATASET_REDUCIDO,
    FIG_DIR
)


def ejecutar_analisis_datos():
    print("\n" + "="*60)
    print("FASE 1 Y 3: PREPROCESAMIENTO Y ANÁLISIS DE CORRELACIÓN")
    print("="*60)

    # 1. Cargar datos
    if not RUTA_DATASET_COMPLETO.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {RUTA_DATASET_COMPLETO}")

    df = pd.read_excel(RUTA_DATASET_COMPLETO)
    print(f"Dimensiones iniciales: {df.shape[0]} filas x {df.shape[1]} columnas")

    # 2. Graficar distribución de clases
    plt.figure(figsize=(8, 4.5))
    sns.countplot(data=df, x="Clase", hue="Clase", palette="crest", legend=False)
    plt.title("Distribución de Clases (20 sujetos x 10 comandos EOG)", fontsize=11, fontweight="bold")
    plt.xlabel("Clase (1 al 10)")
    plt.ylabel("Número de Muestras")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "distribucion_clases.png", dpi=300)
    plt.close()

    # 3. Eliminar Longitud (Prevención de fuga de información)
    if "Longitud" in df.columns:
        df_sin_fuga = df.drop(columns=["Longitud"])
        print("[✓] Columna 'Longitud' eliminada. Previene fuga de información temporal.")
    else:
        df_sin_fuga = df.copy()

    X_df = df_sin_fuga.drop(columns=["Clase"])
    y_serie = df_sin_fuga["Clase"]
    caracteristicas = X_df.columns.tolist()

    # 4. Z-Score manual con NumPy (media=0, std=1)
    X_vals = X_df.to_numpy(dtype=np.float64)
    medias = np.mean(X_vals, axis=0)
    desviaciones = np.std(X_vals, axis=0)
    desviaciones_seguras = np.where(desviaciones == 0, 1e-12, desviaciones)
    X_estandarizado = (X_vals - medias) / desviaciones_seguras

    # 5. Matriz de Correlación de Pearson con NumPy
    N = X_estandarizado.shape[0]
    R_matriz = (X_estandarizado.T @ X_estandarizado) / N

    df_corr = pd.DataFrame(R_matriz, index=caracteristicas, columns=caracteristicas)

    # Mapa de calor de correlación inicial
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        df_corr,
        cmap="vlag",
        center=0,
        vmin=-1.0,
        vmax=1.0,
        annot=False,
        cbar_kws={'label': 'Coeficiente de Pearson (r)'}
    )
    plt.title("Matriz de Correlación Inicial (29 Características EOG)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "matriz_correlacion.png", dpi=300)
    plt.close()

    # 6. Filtrado de pares con |r| > 0.80
    UMBRAL = 0.80
    num_vars = len(caracteristicas)
    pares_altos = []

    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            if np.abs(R_matriz[i, j]) > UMBRAL:
                pares_altos.append((caracteristicas[i], caracteristicas[j], R_matriz[i, j]))

    print(f"\nTotal de pares redundantes (|r| > {UMBRAL}): {len(pares_altos)}")
    for v1, v2, r in pares_altos:
        print(f"  - {v1} <-> {v2}: r = {r:.4f}")

    # Selección voraz sin redundancia: conserva una y descarta la repetida
    indices_conservados = []
    indices_descartados = []

    for i in range(num_vars):
        es_redundante = False
        for c in indices_conservados:
            if np.abs(R_matriz[c, i]) > UMBRAL:
                es_redundante = True
                break
        if es_redundante:
            indices_descartados.append(i)
        else:
            indices_conservados.append(i)

    vars_finales = [caracteristicas[i] for i in indices_conservados]
    vars_descartadas = [caracteristicas[i] for i in indices_descartados]

    print(f"\nVariables iniciales: {len(caracteristicas)}")
    print(f"Variables eliminadas: {len(vars_descartadas)}")
    print(f"Variables conservadas: {len(vars_finales)}")

    # 7. Exportar dataset reducido con valores crudos originales
    df_reducido = X_df[vars_finales].copy()
    df_reducido["Clase"] = y_serie.values

    df_reducido.to_excel(RUTA_DATASET_REDUCIDO, index=False)
    print(f"[✓] Dataset reducido guardado en: {RUTA_DATASET_REDUCIDO}")

    return df_sin_fuga, df_reducido


if __name__ == "__main__":
    ejecutar_analisis_datos()
