# ============================================================
# main.py
# Script Maestro: Ejecuta el flujo completo de principio a fin
# Proyecto: Agrupamiento de Señales EOG del Lenguaje Blink-to-Speak
# ============================================================

import sys
from pathlib import Path

# Agregar src al path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

import importlib

data_module = importlib.import_module("01_data_correlation")
mlp_module = importlib.import_module("02_mlp_experiments")

ejecutar_analisis_datos = data_module.ejecutar_analisis_datos
ejecutar_todos_los_experimentos = mlp_module.ejecutar_todos_los_experimentos


def main():
    print("\n" + "#"*65)
    print("# PROYECTO FINAL: AGRUPAMIENTO DE SEÑALES EOG (BLINK-TO-SPEAK)")
    print("# CUCEI - Aprendizaje Máquina")
    print("#"*65)

    # 1. Fases 1 y 3: Preprocesamiento, Z-score, Correlación y Dataset Reducido
    ejecutar_analisis_datos()

    # 2. Fases 2 y 4: MLP desde cero con NumPy y Validación Cruzada 5-Fold
    ejecutar_todos_los_experimentos()

    print("\n" + "#"*65)
    print("# PROCESO COMPLETADO EXITOSAMENTE")
    print("# Todos los datasets y figuras han sido generados.")
    print("#"*65 + "\n")


if __name__ == "__main__":
    main()
