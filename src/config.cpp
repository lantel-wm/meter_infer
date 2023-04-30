#include <string>
#include "config.hpp"

const std::string ENGINE_PATH = "/home/zzy/meter_infer/engines/";
const std::string IMAGE_PATH = "/home/zzy/meter_infer/data/images/";
const std::string VIDEO_PATH = "/home/zzy/meter_infer/data/videos/";
const std::string LOG_PATH = "/home/zzy/meter_infer/logs/";

const char* INPUT_NAME = "images";
const char* OUTPUT_NAME0 = "output0";
const char* OUTPUT_NAME1 = "output1";

// input tensor size: [-1, 3, 640, 640]
const int BATCH_SIZE = 1;
const int IN_CHANNEL = 3;
const int IN_WIDTH = 640;
const int IN_HEIGHT = 640;

// detection output0 tensor size: [-1, 7, 8400]
const int DET_OUT_CHANNEL0 = 7;
const int DET_OUT_CHANNEL1 = 8400;

// segmentation output0 tensor size: [-1, 38, 8400]
const int SEG_OUT0_CHANNEL0 = 38;
const int SEG_OUT0_CHANNEL1 = 8400;

// segmentation output1 tensor size: [-1, 32, 160, 160]
const int SEG_OUT1_CHANNEL0 = 32;
const int SEG_OUT1_CHANNEL1 = 160;
const int SEG_OUT1_CHANNEL2 = 160;

const int METER = 0; // pressure meter
const int WATER = 1; // water level gauge
const int LEVEL = 2; // water level