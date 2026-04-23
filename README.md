## LVS Take-Home Technical Problems ##

### Task 1 ### 

#### The branch *feature/inference-best.pt* - contains the model chosen for an edge deployment scenario with a markdown file explaining why it was selected. A benchmark script for this model is here also. ####
```text
- models/best.pt
- benchmark/benchmark_inference.py
```
#### The branch *feature/inference-onnx* - contains the converted ONNX model file and a benchmarking script.####
```text
- models/parking_detector.onnx
- benchmark/benchmark_onnx_inference.py
```
#### Table of Benchmarking Results ####

|                        | Before Optimization | After Optimization |
|:----------------------:|:-------------------:|:------------------:|
| Throughput    (img/sec)| 11.77               | 13.24              |
| Avg latency   (ms)     | 80.40               | 68.58              |
| Min latency   (ms)     | 60.31               | 38.29              |
| Max latency   (ms)     | 238.17              | 206.68             |
| StDev latency (ms)     | 10.93               | 10.64              |
| Total Time    (sec)    | 210.95              | 187.49             |
| Images processed       | 2483                | 2483               | 
