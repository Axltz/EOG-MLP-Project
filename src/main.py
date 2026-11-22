import sys
from pathlib import Path
import importlib

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

data_module = importlib.import_module("01_data_correlation")
mlp_module = importlib.import_module("02_mlp_experiments")

ejecutar_analisis_datos = data_module.ejecutar_analisis_datos
ejecutar_todos_los_experimentos = mlp_module.ejecutar_todos_los_experimentos


def main():
    print("Ejecutando Fase 1 y 3...")
    ejecutar_analisis_datos()
    print("Ejecutando Fase 2 y 4...")
    ejecutar_todos_los_experimentos()
    print("Proceso completado exitosamente.")


if __name__ == "__main__":
    main()
