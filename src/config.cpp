#include <string>
#include "config.hpp"

const std::string PROJECT_PATH = "/home/zzy/meter_infer/";
const std::string ENGINE_PATH = "/home/zzy/meter_infer/engines/";
const std::string IMAGE_PATH = "/home/zzy/meter_infer/data/images/";
const std::string VIDEO_PATH = "/home/zzy/meter_infer/data/videos/";
const std::string LOG_PATH = "/home/zzy/meter_infer/logs/";
const std::string DEBUG_PATH = "/home/zzy/reading_images/";

const char* INPUT_NAME = "images";
const char* OUTPUT_NAME0 = "output0";
const char* OUTPUT_NAME1 = "output1";

// class names
const std::vector<std::string> CLASS_NAMES = {"meter", "water", "level"};
const std::vector<std::string> CLASS_NAMES2 = {"pointer", "scale"};

// input tensor size: [-1, 3, 640, 640]
const int CLASS_NUM = 3;
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

// detection confidence threshold
const float CONF_THRESH = 0.25;
// detection iou threshold
const float NMS_THRESH = 0.45;

// warmup time
const int WARMUP_TIME = 10;

const int METER = 0; // pressure meter
const int WATER = 1; // water level gauge
const int LEVEL = 2; // water level

// colors for drawing boxes
const std::vector<cv::Scalar> COLORS = {
    cv::Scalar(201, 207, 142),
    cv::Scalar(122, 190, 255),
    cv::Scalar(111, 127, 250)
};

const int RECT_WIDTH = 512; // 512 pixels, 360 / 512 degrees per pixel
const int RECT_HEIGHT = 32;
const int CIRCLE_WIDTH = 160; // 160 pixels
const int CIRCLE_HEIGHT = 160;

extern const float METER_RANGES[2] = {4.0f, 100.0f};
extern const char METER_UNITS[2][50] = {"kPa", "%"};