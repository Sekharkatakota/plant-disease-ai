import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
VIT_WEIGHTS = os.getenv("VIT_WEIGHTS", str(BASE / "models/vit_plant.pt"))
KERAS_MODEL = os.getenv("KERAS_MODEL", str(BASE / "models/plant_disease_recog_model_pwp.keras"))
CLASSIFIER_BACKEND = os.getenv("CLASSIFIER_BACKEND", "auto")  # auto | vit | keras
CLASS_NAMES = os.getenv("CLASS_NAMES", str(BASE / "models/class_names.json"))
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", str(BASE / "models/yolo_lesion.pt"))  # optional
SOLUTIONS = os.getenv("SOLUTIONS", str(BASE / "data/solutions.json"))
MAX_UPLOAD_MB = 8
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.60"))  # below this -> "uncertain"
