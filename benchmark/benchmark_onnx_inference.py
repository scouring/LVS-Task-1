from pathlib import Path
import time
import statistics

from app.inference_service import ParkingInferenceService
from app.preprocess import preprocess_image


MODEL_PATH = "models/parking_detector.onnx"
INPUT_DIR = Path("data/validation_images")


def benchmark(num_warmup: int = 5):
    """
    Benchmark ONNX image inference.

    Reports:
        throughput (images/sec)
        avg latency (ms)
        min/max/std latency
    """

    service = ParkingInferenceService(MODEL_PATH)

    image_paths = sorted(INPUT_DIR.glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {INPUT_DIR}"
        )

    print("\n===== ONNX BENCHMARK START =====\n")
    print(f"Images found: {len(image_paths)}")
    print(f"Warmup runs: {num_warmup}\n")

    # --------------------------------
    # Warmup
    # --------------------------------
    warmup_tensor, _ = preprocess_image(str(image_paths[0]))

    for _ in range(num_warmup):
        _ = service.infer(warmup_tensor)

    print("Warmup complete.\n")

    # --------------------------------
    # Timed benchmark
    # --------------------------------
    latencies_ms = []

    total_start = time.perf_counter()

    for image_path in image_paths:
        tensor, _ = preprocess_image(str(image_path))

        start = time.perf_counter()
        _ = service.infer(tensor)
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

    std_latency = (
        statistics.stdev(latencies_ms)
        if len(latencies_ms) > 1
        else 0
    )

    # --------------------------------
    # Results
    # --------------------------------
    print("===== ONNX BENCHMARK RESULTS =====")
    print(f"Images processed      : {num_images}")
    print(f"Total time (sec)      : {total_time_sec:.4f}")
    print(f"Throughput (img/sec)  : {throughput:.2f}")
    print(f"Avg latency (ms)      : {avg_latency:.2f}")
    print(f"Min latency (ms)      : {min_latency:.2f}")
    print(f"Max latency (ms)      : {max_latency:.2f}")
    print(f"Std latency (ms)      : {std_latency:.2f}")
    print("==================================\n")


if __name__ == "__main__":
    benchmark()