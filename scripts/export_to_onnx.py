from ultralytics import YOLO
from pathlib import Path


MODEL_PATH = Path("models/best.pt")
OUTPUT_DIR = Path("models")


def export_model():
    model = YOLO(MODEL_PATH)

    model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        dynamic=False,
        simplify=True,
        nms=True
    )

    print("ONNX export complete.")


if __name__ == "__main__":
    export_model()