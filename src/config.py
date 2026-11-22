from pathlib import Path

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "fig"
REPORTE_DIR = BASE_DIR / "reporte"

RUTA_DATASET_COMPLETO = DATA_DIR / "DataSet_completo(1).xlsx"
RUTA_DATASET_REDUCIDO = DATA_DIR / "Dataset_reducido.xlsx"

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Parámetros globales
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 350
LEARNING_RATE = 0.01
BATCH_SIZE = 16

# Espacio de búsqueda de configuraciones MLP
CONFIGURACIONES_MLP = [
    {
        "nombre": "Arquitectura_Pequena",
        "capas_ocultas": [32],
        "activacion": "relu",
        "lr": 0.01,
        "epochs": 350
    },
    {
        "nombre": "Arquitectura_Mediana",
        "capas_ocultas": [64, 32],
        "activacion": "relu",
        "lr": 0.01,
        "epochs": 350
    },
    {
        "nombre": "Arquitectura_Profunda",
        "capas_ocultas": [64, 32, 16],
        "activacion": "tanh",
        "lr": 0.005,
        "epochs": 400
    }
]
