<h1 align="center">LVS Task 1 – Baseline Model - pre-optimization</h1>

<p align="center">
  YOLOv8-based parking space detection with PyTorch and ONNX inference pipelines,
  benchmarking utilities, and edge deployment readiness. Trained with the PKLot
  dataset.
</p>

---

## Overview ##

This project implements an end-to-end **object detection inference pipeline** for parking space occupancy detection using a **YOLOv8 model trained on the PKLot dataset**.

The repository includes:

- PyTorch inference using `best.pt`
- output annotation and prediction saving
- benchmarking scripts for latency / throughput analysis & accuracy
- deployment-ready project scaffolding

The primary objective is to support **efficient inference and benchmarking for edge deployment workflows**.

---

#### Project Structure ####

```text
LVS-Task-1/
│
├── benchmark/
│   ├── __init__.py
│   ├── accuracy_best.py
│   └── benchmark_inference.py
│
├── configs/
│   └── config.py
|
├── inference/
│   ├── yolo_infer.py
│
├── models/
│   ├── best.pt
│
├── data/
|   ├── validation_images/
│   └── validation_labels/
│
├── output/
│   └── predictions/
|
├── pipeline/
│   ├── run_batch_inference.py
│
├── utils/
│   ├── io_utils.py
│   └── viz_utils.py
├── .gitignore
├── requirements.txt
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
git clone -b feature/inference-best.pt https://github.com/scouring/LVS-Task-1.git
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
python -m pipeline.run_batch_inference
```
```text
This will:
- process validation images
- run detection
- save annotated outputs in output/predictions
```

### Benchmarking Workflow ###
```bash
python -m benchmark.benchmark_inference.py
```
```text
This will calculate:
- latency
- FPS
- throughput
```
```bash
python -m benchmark.accuracy_best.py
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




