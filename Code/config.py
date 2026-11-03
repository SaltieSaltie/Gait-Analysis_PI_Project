import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "Models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n-pose.pt")
CONF_THRESHOLD = 0.5

# Setari CAMERA
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30.0

# Setari pentru textul de stare
STATUS_COLORS = {
    "LOADING": (0, 165, 255), # Orange
    "READY": (0, 255, 0), # Green
    "CAPTURING": (0, 0, 255), # Red
    "LOG SAVED": (255, 255, 0)  # Cyan
}
LOG_SAVED_DURATION_SEC = 3.0 # durata in care sta LOG SAVED pe ecran dupa ce a terminat omul mersul