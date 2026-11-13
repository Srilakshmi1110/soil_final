import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SERIAL_PORT = "COM10"
SERIAL_BAUD = 9600

FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000

SECRET_KEY = os.environ.get(
    "SOILAI_SECRET_KEY",
    "soilai-local-development-key"
)

BACKEND_DIR = os.path.join(BASE_DIR, "backend")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

MODEL_DIR = os.path.join(BACKEND_DIR, "models")
DATA_DIR = os.path.join(BACKEND_DIR, "data")

DATABASE_PATH = os.path.join(
    DATA_DIR,
    "predictions.db"
)