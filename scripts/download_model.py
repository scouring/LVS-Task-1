#!/usr/bin/env python3
"""
download_model.py — Download pre-trained YOLOv5n ONNX model for C++ inference.

Usage:
    python3 scripts/download_model.py --model yolov5n  # YOLOv5 nano

Requirements:
    pip install ultralytics          # for YOLOv5
    pip install torch torchvision    # optional: faster export
"""

import argparse
import sys
import os

MODEL_URLS = {
    # Direct ONNX downloads (no Python deps needed)
    "yolov5n": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5nu.pt"
}

def download_yolov5n_onnx(model_name: str, output_dir: str, img_size: int = 640):
    """Download .pt and export to ONNX via ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.\n  pip install ultralytics")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")

    if os.path.exists(onnx_path):
        print(f"[✓] Model already exists: {onnx_path}")
        return onnx_path

    print(f"[→] Loading {model_name} (auto-downloads .pt if needed)…")
    model = YOLO(f"{model_name}.pt")   # downloads weights automatically

    print(f"[→] Exporting to ONNX (imgsz={img_size}, opset=12)…")
    export_path = model.export(
        format="onnx",
        imgsz=img_size,
        opset=12,
        simplify=True,
        dynamic=False,
    )

    # Move to our models dir
    import shutil
    shutil.copy(export_path, onnx_path)
    print(f"[✓] ONNX model saved to: {onnx_path}")
    return onnx_path


def download_coco_names(output_dir: str):
    """Download COCO class names file."""
    import urllib.request
    url = ("https://raw.githubusercontent.com/ultralytics/ultralytics"
           "/main/ultralytics/cfg/datasets/coco.yaml")
    dest = os.path.join(output_dir, "coco.names")

    if os.path.exists(dest):
        print(f"[✓] Classes file already exists: {dest}")
        return dest

    print("[→] Downloading COCO class names…")
    try:
        import urllib.request, yaml
        with urllib.request.urlopen(url) as r:
            data = yaml.safe_load(r.read())
        with open(dest, "w") as f:
            f.write("\n".join(data["names"].values()))
        print(f"[✓] Class names saved to: {dest}")
    except Exception:
        # Fallback: write inline COCO names
        COCO = [
            "person","bicycle","car","motorcycle","airplane","bus","train","truck",
            "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
            "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
            "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
            "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
            "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
            "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
            "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
            "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
            "toaster","sink","refrigerator","book","clock","vase","scissors",
            "teddy bear","hair drier","toothbrush",
        ]
        with open(dest, "w") as f:
            f.write("\n".join(COCO))
        print(f"[✓] Class names (inline COCO) saved to: {dest}")
    return dest


def main():
    parser = argparse.ArgumentParser(description="Download YOLO ONNX model")
    parser.add_argument("--model", default="yolov5n",
                        choices=list(MODEL_URLS.keys()),
                        help="Which model to download (default: yolov8n)")
    parser.add_argument("--output-dir", default="models",
                        help="Directory to save model files (default: models/)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size for ONNX export (default: 640)")
    args = parser.parse_args()

    print(f"\n{'─'*50}")
    print(f"  YOLO Model Downloader")
    print(f"  Model   : {args.model}")
    print(f"  Output  : {args.output_dir}/")
    print(f"  Img size: {args.imgsz}")
    print(f"{'─'*50}\n")

    onnx_path = download_yolov5n_onnx(args.model, args.output_dir, args.imgsz)
    names_path = download_coco_names(args.output_dir)

    print(f"\n{'─'*50}")
    print("  Ready! Run inference with:")
    print(f"  ./build/yolo_inference \\")
    print(f"      --model  {onnx_path} \\")
    print(f"      --classes {names_path} \\")
    print(f"      --video  path/to/video.mp4")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    main()
