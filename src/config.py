import os

# Rutas básicas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Nombre de archivo por defecto
default_csv_name = "crime_demo.csv"
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, default_csv_name)

# Semilla global
RANDOM_SEED = 42

# Columna esperadas mínimas
REQUIRED_COLUMNS_MIN = ["district", "date", "count"]

# Configuración de lags y ventanas
LAG_STEPS = [1, 2, 3, 4]
ROLLING_WINDOW = 4

# Parámetros por defecto de entrenamiento
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 32
DEFAULT_TASK = "classification"  # o "poisson"

# Parámetros del modelo
DEFAULT_HIDDEN_UNITS = [32, 16]
DEFAULT_DROPOUT = 0.1

# Configuración de Recall@K hotspots
DEFAULT_K_HOTSPOTS = 10
