# Meter Reader Inference

## Introduction

This repository contains the code for the inference of the meter reader project. The meter reader project is a project that aims to read the pointer pressure gauge from images, videos or RSTP streams. The project is divided into two parts: the inference and postprocess.

The inference part is implemented by trained detection model and segmentation model. The model is trained on a dataset of images of pressure gauges. We are currently using YOLOv8 model.

The postprocess part contains a computer vision algorithm based on OpenCV. The algorithm reads the value of the pressure gauge from the inference result.

File structure:

```bash
.
├── bin
├── CMakeLists.txt
├── data
│   ├── images
│   └── videos
├── engine
│   ├── yolov8n_batch8.trt
│   ├── yolov8n_fp16.trt
│   ├── yolov8n_int8.trt
│   ├── yolov8s-seg_fp16.trt
│   ├── yolov8s-seg_int8.trt
│   └── yolov8s-seg.trt
├── include
│   ├── config.hpp
│   ├── meter_reader.hpp
│   └── stream_to_img.hpp
├── README.md
└── src
    ├── main.cpp
    ├── meter_reader.cpp
    └── stream_to_img.cpp
```

## Requirements

- gcc 9.4.0
- cmake 3.26.3
- Python 3.10.9
- OpenCV 4.7.0
- cuda 11.4
- cudnn 8.9.0
- TensorRT 8.6.0.12
- glog 0.7.0

## Usage

clone the repository

```bash
git clone https://github.com/ZZY000926/meter_infer.git
cd meter_infer
```

put the serialized TensorRT model in the `engine` folder

```bash
engine
├── yolov8n.trt
├── yolov8n_fp16.trt
├── yolov8n_int8.trt
├── yolov8s-seg.trt
├── yolov8s-seg_fp16.trt
└── yolov8s-seg_int8.trt
```

put the images or videos in the `data` folder

```bash
data
├── images
└── videos
```

build the project

```bash
mkdir build && cd build
cmake ..
make
```

run the project

```bash
./meter_infer
```



