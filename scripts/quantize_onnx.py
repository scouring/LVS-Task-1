from onnxruntime.quantization import quantize_dynamic, QuantType


FP32_MODEL = "models/parking_detector.onnx"
INT8_MODEL = "models/parking_detector_int8.onnx"


def quantize_model():
    quantize_dynamic(
        model_input=FP32_MODEL,
        model_output=INT8_MODEL,
        weight_type=QuantType.QInt8
    )

    print("INT8 quantized ONNX model saved.")


if __name__ == "__main__":
    quantize_model()