# Summary

dir tree:

```
.
├── bin
├── CMakeLists.txt
├── data
│   ├── images
│   │   ├── 123.png
│   │   ├── 60.png
│   │   ├── 61.jpg
│   │   └── 62.jpg
│   └── videos
│       └── 201.mp4
├── device_info.txt
├── engines
│   ├── yolov8n_batch1.trt
│   ├── yolov8n_batch8.trt
│   ├── yolov8n_int8.trt
│   ├── yolov8s-seg_batch8.trt
│   ├── yolov8s-seg_fp16.trt
│   ├── yolov8s-seg_int8.trt
│   └── yolov8s-seg.trt
├── include
│   ├── common.hpp
│   ├── config.hpp
│   ├── detect.hpp
│   ├── meter_reader.hpp
│   ├── segment.hpp
│   └── stream.hpp
├── LICENSE
├── README.md
└── src
    ├── config.cpp
    ├── detect.cpp
    ├── main.cpp
    ├── meter_reader.cpp
    ├── preprocess.cu
    └── stream.cpp

```

## Workflow:

- training(in training server, GTX1080Ti):

    -> weights file (.pt)

- yolov8 export:

    .pt file -> .onnx file (dynamic batch)

- trtexec(in infer server, Tesla P4):

    .onnx file -> .trt engine

    trtexec config:
    ~~~
    trtexec --onnx=weights/yolov8n_dynamic.onnx  --saveEngine=yolov8n_batch8.trt --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:8x3x640x640    
    trtexec --onnx=weights/yolov8s-seg_dynamic.onnx  --saveEngine=yolov8s-seg_batch8.trt --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:8x3x640x640
    Average on 10 runs - GPU latency: 78.9045 ms - Host latency: 85.9525 ms (enqueue 2.52272 ms)
    ~~~

- meter_reader main program (cpp)

## yolov8-seg

input:

- images: 1x3x640x640

output:

- output0: 8x38x8400

    8: batch size
    38 = xc + yc + w + h + 2 class confidence + 32 mask weights

- output1: 8x32x160x160
    prototype masks

multiply each mask with its corresponding mask weight and then sum all these products to get the final mask

## TODO:

- int8 inference 

- segmentation (done)

- cuda preprocess and postprocess (cuda high performance multi-dimensional array transpose needed) (done)

- meter_reader (done)

- multi-thread to read RTSP stream (done)

- producer-consumer model (video stream as producer, meter_reader as consumer) (done)

- save real-time readings in mysql server

- use ZLMediaKit