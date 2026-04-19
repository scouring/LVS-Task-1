from pathlib import Path
import cv2

from app.inference_service import ParkingInferenceService
from app.preprocess import preprocess_image
from app.postprocess import annotate_results


MODEL_PATH = "models/parking_detector.onnx"
INPUT_DIR = Path("data/validation_images")
OUTPUT_DIR = Path("output/predictions")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_batch_inference():
    service = ParkingInferenceService(MODEL_PATH)

    for image_path in INPUT_DIR.glob("*.jpg"):
        tensor, original = preprocess_image(str(image_path))

        outputs = service.infer(tensor)

        detections = outputs[0][0]

        result_image, occupied, empty = annotate_results(
            original,
            detections
        )

        save_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(save_path), result_image)

        print(
            f"{image_path.name}: occupied={occupied}, empty={empty}"
        )


if __name__ == "__main__":
    run_batch_inference()