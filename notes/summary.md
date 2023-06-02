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
│   └── stream_to_img.hpp
├── LICENSE
├── README.md
└── src
    ├── config.cpp
    ├── detect.cpp
    ├── main.cpp
    ├── meter_reader.cpp
    ├── preprocess.cu
    └── stream_to_img.cpp

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
    trtexec --onnx=weights/yolov8n_dynamic.onnx  --saveEngine=yolov8n_int8.trt --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:8x3x640x640 --int8 
    ~~~

- meter_reader main program (cpp)

## TODO:

- int8 inference

- segmentation

- cuda preprocess and postprocess (cuda high performance multi-dimensional array transpose needed)

- meter_reader

- multi-thread to read RTSP stream

- producer-consumer model (video stream as producer, meter_reader as consumer)