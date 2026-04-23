from ultralytics import YOLO
from configs.config import MODEL_PATH, IMG_SIZE, CONF_THRESHOLD


class YOLOInferencer:
    def __init__(self):
        self.model = YOLO(str(MODEL_PATH))

    def predict(self, image):
        """
        Runs inference on a single image.
        Returns raw YOLO results.
        """
        results = self.model.predict(
            source=image,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            verbose=False
        )
        return results[0]