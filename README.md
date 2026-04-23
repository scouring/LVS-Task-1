<h1 align="center">LVS Task 2 – ONNX Model - Quantized to FP16</h1>

<p align="center">
  YOLOv8-based parking space detection with PyTorch and ONNX inference pipelines,
  benchmarking utilities, and edge deployment readiness. Trained with the PKLot
  dataset. The model for inference in this branch has been optimized to an FP16 format.
</p>

---

## Overview ##

This project implements an end-to-end **object detection inference pipeline** for parking space occupancy detection using a **YOLOv8 model trained on the PKLot dataset**.
This branch has an optimized ONNX FP16 model for edge deployment.

The repository includes:

- Inference using `parking_detector_fp16.onnx`
- output annotation and prediction saving
- benchmarking scripts for latency / throughput analysis & accuracy
- deployment-ready project scaffolding

The primary objective is to support **efficient inference and benchmarking for edge deployment workflows**.

---

#### Project Structure ####

```text
LVS-Task-1/
├── app/
|   ├── config.py
|   ├── inference_service.py
|   ├── main.py
|   ├── postprocess.py
|   ├── preprocess.py
│   └── utils.py
│
├── benchmark/
│   ├── accuracy_fp16.py
│   └── benchmark_fp16_inference.py
|
├── models/
│   ├── best.pt
│   ├── export_to_onnx.py
│   ├── parking_detector.onnx
│   └── parking_detector_fp16.onnx
│
├── data/
|   ├── validation_images/
│   └── validation_labels/
│
├── output/
│   └── predictions/
|
├── scripts/
|   ├── convert_to_fp16.py
│   └── export_to_onnx.py
│
├── .gitignore
├── requirements.txt
├── run_edge_inference.py
└── README.md
```
---

```text
This model uses a large validation dataset you will need. It can be accessed and put into your cloned repo, as above - under **data**, from these links:
- validation_images: https://drive.google.com/drive/folders/1JceN4PEB8R7KhnijBKpXD95bh0O79r7B?usp=drive_link
- validation_labels: https://drive.google.com/drive/folders/13q1kGRLvvqG4Wyk-z_22E6W9NbMyuMM5?usp=drive_link
```

---

#### Installation ####

#### 1. Clone the repository ####
```bash
git clone -b feature/inference-fp16 https://github.com/scouring/LVS-Task-1.git
cd LVS-Task-1
```
#### 2. Download the validation images and labels ####
```bash
- validation_images: https://drive.google.com/drive/folders/1JceN4PEB8R7KhnijBKpXD95bh0O79r7B?usp=drive_link
- validation_labels: https://drive.google.com/drive/folders/13q1kGRLvvqG4Wyk-z_22E6W9NbMyuMM5?usp=drive_link

Place these in the project structure under "data" (as shown above)
```
#### 3. Create a folder at the base of the project for the output/predictions
```bash
As shown in the project structure above:
├── output/
│   └── predictions/
```
#### 4. Create a virtual environment ####
```bash
python3 -m venv .venv
source .venv/bin/activate
```
#### 5. Install dependencies ####
```bash
pip install -r requirements.txt
```

---

### WORKFLOW ###

#### Run PyTorch Inference ####
```bash
python run_batch_inference.py
```
```text
This will:
- process validation images
- run detection
- save annotated outputs in output/predictions
```

### Benchmarking Workflow ###
```bash
python -m benchmark.benchmark_fp16_inference
```
```text
This will calculate:
- latency
- FPS
- throughput
```
```bash
python -m benchmark.accuracy_int8
```
```text
This will calculate accuracy metrics:
- True Positives (TP)
- False Positives (FP)
- False Negatives (FN)
```

### Author ###
```text
Sherry Courington
AI/Computer Vision Engineer
Visual Computing | Edge AI | MLOps
```




