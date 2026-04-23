from pathlib import Path
import numpy as np

from configs.config import INPUT_IMAGES_DIR, CONF_THRESHOLD, CLASS_NAMES
from inference.yolo_infer import YOLOInferencer
from utils.io_utils import load_image


LABELS_DIR = INPUT_IMAGES_DIR.parent / "validation_labels"  # YOLO .txt label files
IOU_THRESHOLD = 0.5


def load_ground_truth(label_path: Path, img_h: int, img_w: int) -> np.ndarray:
    """
    Load YOLO-format label file and convert to absolute [x1, y1, x2, y2, cls].
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
    inferencer = YOLOInferencer()

    image_paths = sorted(INPUT_IMAGES_DIR.glob("*.*"))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {INPUT_IMAGES_DIR}")

    all_tp = 0
    all_fp = 0
    all_fn = 0

    # Per-class counters: {cls_id: {"tp": 0, "fp": 0, "fn": 0}}
    per_class = {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(len(CLASS_NAMES))}

    per_image_results = []

    print("\n===== ACCURACY EVALUATION START =====\n")
    print(f"Images found    : {len(image_paths)}")
    print(f"Labels dir      : {LABELS_DIR}")
    print(f"Conf threshold  : {CONF_THRESHOLD}")
    print(f"IoU threshold   : {IOU_THRESHOLD}\n")

    for image_path in image_paths:
        image = load_image(image_path)

        if image is None:
            print(f"[WARN] Could not load {image_path.name}, skipping.")
            continue

        img_h, img_w = image.shape[:2]

        results = inferencer.predict(image)

        # Extract predictions above conf threshold (already filtered by YOLOInferencer)
        preds = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            preds.append([x1, y1, x2, y2, conf, cls])

        label_path = LABELS_DIR / (image_path.stem + ".txt")
        gt_boxes = load_ground_truth(label_path, img_h, img_w)

        matched_gt = set()
        tp = fp = 0

        for pred in preds:
            pred_box = pred[:4]
            pred_cls = pred[5]

            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                if int(gt[4]) != pred_cls:  # class must match
                    continue
                iou = compute_iou(pred_box, gt[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= IOU_THRESHOLD and best_gt_idx >= 0:
                tp += 1
                matched_gt.add(best_gt_idx)
                per_class[pred_cls]["tp"] += 1
            else:
                fp += 1
                per_class[pred_cls]["fp"] += 1

        fn = len(gt_boxes) - len(matched_gt)

        # Attribute FNs to the class of the unmatched GT boxes
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx not in matched_gt:
                per_class[int(gt[4])]["fn"] += 1

        all_tp += tp
        all_fp += fp
        all_fn += fn

        per_image_results.append({
            "image": image_path.name,
            "tp": tp, "fp": fp, "fn": fn
        })

    # -------------------------
    # Overall metrics
    # -------------------------
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall    = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print("===== ACCURACY RESULTS (OVERALL) =====")
    print(f"Images evaluated : {len(per_image_results)}")
    print(f"True Positives   : {all_tp}")
    print(f"False Positives  : {all_fp}")
    print(f"False Negatives  : {all_fn}")
    print(f"Precision        : {precision:.4f}")
    print(f"Recall           : {recall:.4f}")
    print(f"F1 Score         : {f1:.4f}")
    print("=======================================\n")

    # -------------------------
    # Per-class metrics
    # -------------------------
    print("===== ACCURACY RESULTS (PER CLASS) =====")
    for cls_id, name in enumerate(CLASS_NAMES):
        c = per_class[cls_id]
        c_tp, c_fp, c_fn = c["tp"], c["fp"], c["fn"]
        c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        c_rec  = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        c_f1   = (2 * c_prec * c_rec / (c_prec + c_rec)
                  if (c_prec + c_rec) > 0 else 0.0)
        print(f"  [{name}]")
        print(f"    TP: {c_tp}  FP: {c_fp}  FN: {c_fn}")
        print(f"    Precision : {c_prec:.4f}")
        print(f"    Recall    : {c_rec:.4f}")
        print(f"    F1        : {c_f1:.4f}")
    print("=========================================\n")


if __name__ == "__main__":
    evaluate()