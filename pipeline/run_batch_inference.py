from pathlib import Path
import cv2

from configs.config import INPUT_IMAGES_DIR, OUTPUT_DIR
from inference.yolo_infer import YOLOInferencer
from utils.io_utils import load_image, save_image
from utils.viz_utils import draw_detections


def run():
    inferencer = YOLOInferencer()

    image_paths = sorted(INPUT_IMAGES_DIR.glob("*.*"))

    print("\n===== PARKING LOT INFERENCE RESULTS =====\n")

    for img_path in image_paths:
        image = load_image(img_path)

        results = inferencer.predict(image)

        annotated, occupied, empty = draw_detections(image, results)

        output_path = OUTPUT_DIR / img_path.name
        save_image(output_path, annotated)

        print(f"{img_path.name} -> Occupied: {occupied} | Empty: {empty}")

    print("\nSaved annotated images to:", OUTPUT_DIR)


if __name__ == "__main__":
    run()