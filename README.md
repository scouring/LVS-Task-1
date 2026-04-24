<h1 align="center">LVS Task 3 – Object Detection Video Demo - YOLO Video Inference — C++</h1>

<p align="center">
  Real-time object detection on video using YOLOv5nu ONNX model and **OpenCV DNN**.  
  No PyTorch, no CUDA required (CPU works out of the box; CUDA optional).
</p>

---

#### Installation ####

#### 1. Clone the repository ####
```bash
git clone -b feature/yolo-cpp https://github.com/scouring/LVS-Task-1.git
cd LVS-Task-1
```

#### Project Structure ####

```text
yolo_cpp/
├── src/
│   └── yolo_inference.cpp   # Main C++ inference engine
├── scripts/
│   └── download_model.py    # Downloads & exports ONNX model
├── models/   
├── videos/                  
├── CMakeLists.txt
└── README.md
```

---

#### 1 — Prerequisites

#### Ubuntu / Debian
```bash
sudo apt update
sudo apt install -y cmake g++ libopencv-dev
# Optional CUDA support:
# sudo apt install -y libopencv-dev  (build OpenCV with CUDA yourself, or use GPU image)
```

---

#### 2 — Download Pre-Trained Model

```bash
pip install ultralytics                             # one-time install
python3 scripts/download_model.py --model yolov5n   # small  (~5 MB)
```

This exports `models/yolov5n.onnx` and `models/coco.names` ready for C++.

---

#### 3 - Download a video, create a videos & models folder
```text
The video I used is 7MB. You can access it here: https://drive.google.com/drive/folders/1eIZF-6hF_tnmx6JSEw9ZDspftaeHAr9p?usp=drive_link. After download, create a videos folder and a models folder at the root of this project (as in Project Structure above). Place the video inside.
```

#### 4 — Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# Binary: build/yolo_inference
```

---

#### 5 — Run

```bash
# On a video file
./build/yolo_inference \
    --model   models/yolov5nu.onnx \
    --video   /path/to/video.mp4 \
    --classes models/coco.names

# Webcam (device 0)
./build/yolo_inference \
    --model models/yolov5nu.onnx \
    --video 0

# Save annotated output + no window (headless / server)
./build/yolo_inference \
    --model   models/yolov5nu.onnx \
    --video   input.mp4 \
    --classes models/coco.names \
    --save    output.mp4 \
    --noshow

# CUDA acceleration
./build/yolo_inference \
    --model  models/yolov5nu.onnx \
    --video  input.mp4 \
    --cuda
```

---

#### 6 — All CLI Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | path | **required** | ONNX model file |
| `--video` | path/int | **required** | Video file or `0` for webcam |
| `--classes` | path | built-in COCO | Newline-delimited class names |
| `--conf` | float | `0.5` | Confidence threshold |
| `--nms` | float | `0.45` | NMS IoU threshold |
| `--width` | int | `640` | Model input width |
| `--height` | int | `640` | Model input height |
| `--save` | path | *(off)* | Write annotated video here |
| `--noshow` | flag | *(off)* | Suppress display window |
| `--cuda` | flag | *(off)* | Use CUDA backend |

**Window controls:** `q` or `ESC` to quit · `Space` to pause

---

### Author ###
```text
Sherry Courington
AI/Computer Vision Engineer
Visual Computing | Edge AI | MLOps
```