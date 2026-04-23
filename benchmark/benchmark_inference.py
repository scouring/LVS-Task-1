from pathlib import Path
import time
import statistics

from configs.config import INPUT_IMAGES_DIR
from inference.yolo_infer import YOLOInferencer
from utils.io_utils import load_image


def benchmark(num_warmup: int = 5):
    """
    Benchmarks YOLOv8 inference on validation images.

    Metrics:
        - throughput: images / sec
        - avg latency: ms / image
        - min/max latency
        - std deviation
    """

    inferencer = YOLOInferencer()

    image_paths = sorted(INPUT_IMAGES_DIR.glob("*.*"))

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {INPUT_IMAGES_DIR}"
        )

    print("\n===== BENCHMARK START =====\n")
    print(f"Images found: {len(image_paths)}")
    print(f"Warmup runs: {num_warmup}\n")

    # -------------------------
    # Warmup
    # -------------------------
    warmup_image = load_image(image_paths[0])

    for _ in range(num_warmup):
        inferencer.predict(warmup_image)

    print("Warmup complete.\n")

    # -------------------------
    # Timed benchmark
    # -------------------------
    latencies_ms = []

    total_start = time.perf_counter()

    for img_path in image_paths:
        image = load_image(img_path)

        start = time.perf_counter()
        _ = inferencer.predict(image)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies_ms.append(latency_ms)

    total_end = time.perf_counter()

    total_time_sec = total_end - total_start
    num_images = len(image_paths)

    throughput = num_images / total_time_sec
    avg_latency = statistics.mean(latencies_ms)
    min_latency = min(latencies_ms)
    max_latency = max(latencies_ms)
    std_latency = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0

    # -------------------------
    # Report
    # -------------------------
    print("===== BENCHMARK RESULTS =====")
    print(f"Images processed      : {num_images}")
    print(f"Total time (sec)      : {total_time_sec:.4f}")
    print(f"Throughput (img/sec)  : {throughput:.2f}")
    print(f"Avg latency (ms)      : {avg_latency:.2f}")
    print(f"Min latency (ms)      : {min_latency:.2f}")
    print(f"Max latency (ms)      : {max_latency:.2f}")
    print(f"Std latency (ms)      : {std_latency:.2f}")
    print("=============================\n")


if __name__ == "__main__":
    benchmark()