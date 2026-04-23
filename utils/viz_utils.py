import cv2
from configs.config import CLASS_NAMES, COLORS


def draw_detections(image, results):
    """
    Draws boxes and returns occupancy counts.
    """

    occupied_count = 0
    empty_count = 0

    boxes = results.boxes

    for box in boxes:
        cls_id = int(box.cls[0])
        label = CLASS_NAMES[cls_id]
        conf = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = COLORS[label]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{label} {conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

        if label == "occupied":
            occupied_count += 1
        else:
            empty_count += 1

    return image, occupied_count, empty_count