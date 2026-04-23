from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx
from pathlib import Path

FP32_MODEL_PATH = Path("models/parking_detector.onnx")
INT8_MODEL_PATH = Path("models/parking_detector_int8.onnx")


def quantize_to_int8():
    print(f"Loading FP32 model: {FP32_MODEL_PATH}")
    onnx.checker.check_model(str(FP32_MODEL_PATH))

    print("Applying dynamic INT8 quantization (weights only)...")
    quantize_dynamic(
        model_input=str(FP32_MODEL_PATH),
        model_output=str(INT8_MODEL_PATH),
        weight_type=QuantType.QUInt8   # weights quantized to uint8
    )

    print("Validating INT8 model...")
    onnx.checker.check_model(str(INT8_MODEL_PATH))
    print("  Model is valid ✓")
    print(f"Saved: {INT8_MODEL_PATH}")


if __name__ == "__main__":
    quantize_to_int8()