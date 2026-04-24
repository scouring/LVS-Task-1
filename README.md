## LVS Take-Home Technical Problems ##

### I. **Task 1 - Model Optimization** ### 

#### The branch *feature/inference-best.pt* - contains the model chosen for an edge deployment scenario. This model was selected from a personal project in my repository because it used YOLOv8 and its metrics (precision & recall) were very good. It also has a large dataset for training, testing, and validating. A benchmark script for this model is also on this branch: ####
```text
- models/best.pt
- benchmark/benchmark_inference.py
```
#### The branch *feature/inference-onnx* - contains the converted ONNX model file and a benchmarking script: ####
```text
- models/parking_detector.onnx
- benchmark/benchmark_onnx_inference.py
```
 #### <h3 align="center"> Table of Benchmarking Results </h3> ####
<div align="center">

|                        | Before Optimization (Best.pt) | After Optimization (ONNX FP32) |
|:----------------------:|:-----------------------------:|:------------------------------:|
| Throughput    (img/sec)| 11.77                         | 13.24                          |
| Avg latency   (ms)     | 80.40                         | 68.58                          |
| Min latency   (ms)     | 60.31                         | 38.29                          |
| Max latency   (ms)     | 238.17                        | 206.68                         |
| StDev latency (ms)     | 10.93                         | 10.64                          |
| Total Time    (sec)    | 210.95                        | 187.49                         |
| Images processed       | 2483                          | 2483                           | 
</div>

### II. **Task 2 Post-Training Quantization** ###

#### <h3 align="center"> The Quantized model artifacts: </h3> ####
```text
- Base model      -> branch: feature/inference-best.pt    file: models/best.pt
- ONNX FP32 model -> branch: feature/inference-onnx       file: models/parking_detector.onnx
- ONNX FP16 model -> branch: feature/inference-fp16       file: models/parking_detector_fp16.onnx
- ONNX INT8 model -> branch: feature/inference-int8       file: models/parking_detector_int8.onnx
```
#### <h3 align="center"> Results Table </h3> ####
<div align="center">

|                        | Best.pt | ONNX FP32 | ONNX FP16 | ONNX INT8 |
|:----------------------:|:-------:|:---------:|:---------:|:---------:|
| Model Size   (MB)      | 6.1     | 12        | 6         | 3.32      |
| Throughput   (img/sec) | 11.77   | 13.24     | 12.23     | 14.10     |
| Avg latency  (ms)      | 80.40   | 68.58     | 69.18     | 64.15     |
| Accuracy     (%)       | 99.51   | 99.60     | 99.6      | 99.54     |

</div>

#### <h3 align="center"> Quantization Recommendation </h3> ####
```text
I established an FP32 baseline, optimized to FP16 for near-lossless acceleration, then evaluated INT8 for maximum edge
throughput. Given the results in the table above, I would recommend the ONNX INT8 model for a power and latency
constrained deployment. The ONNX INT8 model is the smallest of the four models. It also has the greatest throughput
and the least latency. Its accuracy is not substantially lower than ONNX FP16. The FP16 models are usually the best
trade-off between accuracy and latency. But given that this model will be deployed on a small edge device (a drone),
I opt for ONNX INT8 model due to its lesser latency.
```

----

### II. **Task 3 Object Detection Video Demo**

```text
The branch *feature/yolo-cpp* - contains the C++ code to download a YOLOv5 model and the code to convert it to an ONNX model file. Instructions are in its README.
```
----

```text
The input video is a public sample. It came from https://free-stock.video/video/road-city-traffic-street-vehicles-cars-urban-3544#google_vignette.
You may download it here: https://drive.google.com/file/d/1UElgNUtfscP1woD-5Yo2ZxOO0mhyk2lV/view?usp=drive_link
```

```text
The YOLO model variant I used is YOLOv5. I chose it due to its small size (4 MB) and it speed (20 FPS). To run inference on the video, I exported it to an ONNX format. The main function opens the video file and reads metadata. It uses a 'while' loop to pull in one frame at a time. The preprocess function resizes, scales down the frames while maintaining the aspect ration. It also normalizes pixel values. The detect function loads the preprocessed frames into the NN and runs the inference. The postprocess function reads bounding boxes, finds the highest scoring class, converts coordinates to original frame size, and runs Non-Max Suppression (NMS) to eliminate duplicate boxes. The drawDetections function puts a bounding box, confidence score and label around each surrounding detection.
```

----

#### Annotated Video Output - Detection over time.

<p align="center">
  <img src="Recording 2026-04-24 095553.gif" width="700" />
</p>

----

#### Basic Performance ####

```text
The video's top right corner shows the model's latency and frames per second (FPS).
- FPS = ~10.1
- Inf = ~1.8 ms
The only optimizations I used were converting to an ONNX format. I chose YOLOv5 because it is a smaller, faster model (20 FPS) - even on a CPU.
```
