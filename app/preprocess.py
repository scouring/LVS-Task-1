import cv2
import numpy as np


def preprocess_image(image_path: str, img_size: int = 640):
    image = cv2.imread(image_path)
    original = image.copy()

    resized = cv2.resize(image, (img_size, img_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    normalized = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(normalized, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    # FP32 — INT8 dynamic quantization handles casting internally
    return tensor, original