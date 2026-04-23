import cv2
from pathlib import Path


def load_image(image_path):
    return cv2.imread(str(image_path))


def save_image(output_path, image):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)