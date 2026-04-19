import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path


class ParkingInferenceService:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [
            output.name for output in self.session.get_outputs()
        ]

    def infer(self, image_tensor: np.ndarray):
        outputs = self.session.run(
            self.output_names,
            {self.input_name: image_tensor}
        )
        return outputs