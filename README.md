# YOLO Video Inference — C++

Real-time object detection on video using YOLOv8/v5 ONNX models and **OpenCV DNN**.  
No PyTorch, no CUDA required (CPU works out of the box; CUDA optional).

---

## Project Structure

```
yolo_cpp/
├── src/
│   └── yolo_inference.cpp   # Main C++ inference engine
├── scripts/
│   └── download_model.py    # Downloads & exports ONNX model
├── models/                  # ONNX model + class names go here
├── CMakeLists.txt
└── README.md
```

---

## 1 — Prerequisites

### Ubuntu / Debian
```bash
sudo apt update
sudo apt install -y cmake g++ libopencv-dev
# Optional CUDA support:
# sudo apt install -y libopencv-dev  (build OpenCV with CUDA yourself, or use GPU image)
```

### macOS (Homebrew)
```bash
brew install cmake opencv
```

### Windows (vcpkg)
```powershell
vcpkg install opencv4[dnn]:x64-windows
# Add -DCMAKE_TOOLCHAIN_FILE=.../vcpkg/scripts/buildsystems/vcpkg.cmake to cmake
```

---

## 2 — Download Pre-Trained Model

```bash
pip install ultralytics          # one-time install
python3 scripts/download_model.py               # YOLOv8n (~6 MB ONNX)
python3 scripts/download_model.py --model yolov8s   # small  (~22 MB)
python3 scripts/download_model.py --model yolov8m   # medium (~52 MB)
```

This exports `models/yolov8n.onnx` and `models/coco.names` ready for C++.

---

## 3 — Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# Binary: build/yolo_inference
```

---

## 4 — Run

```bash
# On a video file
./build/yolo_inference \
    --model   models/yolov8n.onnx \
    --video   /path/to/video.mp4 \
    --classes models/coco.names

# Webcam (device 0)
./build/yolo_inference \
    --model models/yolov8n.onnx \
    --video 0

# Save annotated output + no window (headless / server)
./build/yolo_inference \
    --model   models/yolov8n.onnx \
    --video   input.mp4 \
    --classes models/coco.names \
    --save    output.mp4 \
    --noshow

# CUDA acceleration
./build/yolo_inference \
    --model  models/yolov8n.onnx \
    --video  input.mp4 \
    --cuda
```

---

## 5 — All CLI Options

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

## 6 — Supported Models

| Model | Size | COCO mAP | Speed (CPU) |
|-------|------|-----------|-------------|
| YOLOv8n | 6 MB | 37.3 | ~15 FPS |
| YOLOv8s | 22 MB | 44.9 | ~8 FPS |
| YOLOv8m | 52 MB | 50.2 | ~3 FPS |
| YOLOv5n | 4 MB | 28.0 | ~20 FPS |

> Speeds are approximate on a modern CPU. CUDA speeds up 5–10×.

---

## 7 — Customising for Your Own Model

1. Export your model to ONNX format at input size 640×640 (or match `--width`/`--height`).
2. Create a plain-text `.names` file with one class per line.
3. Pass both via `--model` and `--classes`.

The inference engine auto-detects YOLOv5 vs YOLOv8 output layout.

---

## Troubleshooting

**`error: OpenCV not found`** — ensure `libopencv-dev` (Linux) or `opencv` (brew) is installed and `pkg-config --modversion opencv4` returns a version.

**Black boxes / wrong detections** — check `--width`/`--height` match the model's training size, and that `--classes` matches the number of classes the model outputs.

**CUDA not available** — rebuild OpenCV with `-DWITH_CUDA=ON`, or omit `--cuda` to run on CPU.
