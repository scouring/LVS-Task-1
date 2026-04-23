from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best.pt"

INPUT_IMAGES_DIR = BASE_DIR / "data" / "val_images"
OUTPUT_DIR = BASE_DIR / "output" / "predictions"

# YOLO settings
IMG_SIZE = 640
CONF_THRESHOLD = 0.25

# Class mapping (must match training order)
CLASS_NAMES = ["empty", "occupied"]

# Colors (BGR format for OpenCV)
COLORS = {
    "empty": (0, 255, 0),      # green
    "occupied": (0, 0, 255),   # red
}