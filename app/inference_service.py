import onnxruntime as ort
import numpy as np
from pathlib import Path


class ParkingInferenceService:
    def __init__(self, model_path: str, use_gpu: bool = False):
        """
        Args:
            model_path: Path to the ONNX model (FP32 or FP16).
            use_gpu:    If True, use CUDA execution provider for
                        native FP16 acceleration. Falls back to CPU.
        """
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            # CPU handles FP16 models by upcasting internally
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [
            output.name for output in self.session.get_outputs()
        ]

        # Log which model and provider are active
        active_provider = self.session.get_providers()[0]
        print(f"[InferenceService] Model     : {model_path}")
        print(f"[InferenceService] Provider  : {active_provider}")
        print(f"[InferenceService] Input     : {self.input_name}")

    def infer(self, image_tensor: np.ndarray) -> list:
        outputs = self.session.run(
            self.output_names,
            {self.input_name: image_tensor}
        )
        return outputs