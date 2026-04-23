from pathlib import Path
import time
import statistics
import numpy as np

from app.inference_service import ParkingInferenceService
from app.preprocess import preprocess_image


MODEL_PATH = "models/parking_detector_fp16.onnx"  # <-- FP16 model
INPUT_DIR = Path("data/validation_images")


def benchmark(num_warmup: int = 5):
    service = ParkingInferenceService(MODEL_PATH)
    image_paths = sorted(INPUT_DIR.glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {INPUT_DIR}")

    print("\n===== FP16 ONNX BENCHMARK START =====\n")
    print(f"Images found : {len(image_paths)}")
    print(f"Warmup runs  : {num_warmup}\n")

    warmup_tensor, _ = preprocess_image(str(image_paths[0]))
    for _ in range(num_warmup):
        service.infer(warmup_tensor)
    print("Warmup complete.\n")

    latencies_ms = []
    total_start = time.perf_counter()

    for image_path in image_paths:
        tensor, _ = preprocess_image(str(image_path))
        start = time.perf_counter()
        service.infer(tensor)
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000)

    total_end = time.perf_counter()

    total_time_sec = total_end - total_start
    num_images = len(image_paths)
    arr = np.array(latencies_ms)

    print("===== FP16 ONNX BENCHMARK RESULTS =====")
    print(f"Images processed      : {num_images}")
    print(f"Total time (sec)      : {total_time_sec:.4f}")
    print(f"Throughput (img/sec)  : {num_images / total_time_sec:.2f}")
    print(f"Avg latency (ms)      : {arr.mean():.2f}")
    print(f"Min latency (ms)      : {arr.min():.2f}")
    print(f"Max latency (ms)      : {arr.max():.2f}")
    print(f"p50 latency (ms)      : {np.percentile(arr, 50):.2f}")
    print(f"p95 latency (ms)      : {np.percentile(arr, 95):.2f}")
    print(f"Std latency (ms)      : {arr.std():.2f}")
    print("========================================\n")


if __name__ == "__main__":
    benchmark()