from pathlib import Path
import numpy as np

from app.inference_service import ParkingInferenceService
from app.preprocess import preprocess_image


MODEL_PATH = "models/parking_detector_fp16.onnx"
INPUT_DIR = Path("data/validation_images")
LABELS_DIR = Path("data/validation_labels")  # YOLO .txt label files
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5


def load_ground_truth(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """
    Load YOLO-format label file and convert to absolute [x1,y1,x2,y2,cls].
    Returns array of shape (N, 5).
    """
    boxes = []
    if not label_path.exists():
        return np.zeros((0, 5))
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, w, h = map(float, parts[:5])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2, cls])
    return np.array(boxes)


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def evaluate():
    service = ParkingInferenceService(MODEL_PATH)
    image_paths = sorted(INPUT_DIR.glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {INPUT_DIR}")

    all_tp = 0
    all_fp = 0
    all_fn = 0
    per_image_results = []

    print("\n===== ACCURACY EVALUATION START =====\n")
    print(f"Images found    : {len(image_paths)}")
    print(f"Labels dir      : {LABELS_DIR}")
    print(f"Conf threshold  : {CONF_THRESHOLD}")
    print(f"IoU threshold   : {IOU_THRESHOLD}\n")

    for image_path in image_paths:
        tensor, original = preprocess_image(str(image_path))
        img_h, img_w = original.shape[:2]

        outputs = service.infer(tensor)
        detections = outputs[0][0].astype(np.float32)  # shape: (N, 6) — x1,y1,x2,y2,conf,cls

        # Scale detections from 640x640 back to original image size
        scale_x = img_w / 640
        scale_y = img_h / 640

        preds = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if conf < CONF_THRESHOLD:
                continue
            preds.append([
                x1 * scale_x, y1 * scale_y,
                x2 * scale_x, y2 * scale_y,
                conf, int(cls)
            ])

        label_path = LABELS_DIR / (image_path.stem + ".txt")
        gt_boxes = load_ground_truth(label_path, img_w, img_h)

        matched_gt = set()
        tp = fp = 0

        for pred in preds:
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                if int(gt[4]) != pred[5]:  # class must match
                    continue
                iou = compute_iou(pred[:4], gt[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= IOU_THRESHOLD and best_gt_idx >= 0:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt)

        all_tp += tp
        all_fp += fp
        all_fn += fn

        per_image_results.append({
            "image": image_path.name,
            "tp": tp, "fp": fp, "fn": fn
        })

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall    = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print("\n===== FP16 ACCURACY RESULTS =====")
    print(f"Images evaluated : {len(image_paths)}")
    print(f"True Positives   : {all_tp}")
    print(f"False Positives  : {all_fp}")
    print(f"False Negatives  : {all_fn}")
    print(f"Precision        : {precision:.4f}")
    print(f"Recall           : {recall:.4f}")
    print(f"F1 Score         : {f1:.4f}")
    print(f"Conf threshold   : {CONF_THRESHOLD}")
    print(f"IoU threshold    : {IOU_THRESHOLD}")
    print("==================================\n")


if __name__ == "__main__":
    evaluate()