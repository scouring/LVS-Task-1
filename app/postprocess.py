import cv2
import numpy as np


def annotate_results(image, detections, conf_threshold=0.5):
    # Cast FP16 outputs to FP32 for stable arithmetic
    detections = detections.astype(np.float32)

    occupied_count = 0
    empty_count = 0

    for det in detections:
        x1, y1, x2, y2, conf, cls = det

        if conf < conf_threshold:
            continue

        label = "occupied" if int(cls) == 1 else "empty"

        if label == "occupied":
            occupied_count += 1
            color = (0, 0, 255)
        else:
            empty_count += 1
            color = (0, 255, 0)

        cv2.rectangle(
            image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            2
        )
        cv2.putText(
            image,
            f"{label}: {conf:.2f}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return image, occupied_count, empty_count