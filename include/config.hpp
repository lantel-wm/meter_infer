#ifndef _CONFIG_HPP_
#define _CONFIG_HPP_

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

extern const std::string ENGINE_PATH;
extern const std::string IMAGE_PATH;
extern const std::string VIDEO_PATH;
extern const std::string LOG_PATH;

extern const char* INPUT_NAME;
extern const char* OUTPUT_NAME0;
extern const char* OUTPUT_NAME1;

extern const std::vector<std::string> CLASS_NAMES; 

extern const int CLASS_NUM;
extern const int BATCH_SIZE;
extern const int IN_CHANNEL;
extern const int IN_WIDTH;
extern const int IN_HEIGHT;

extern const int DET_OUT_CHANNEL0;
extern const int DET_OUT_CHANNEL1;

extern const int SEG_OUT0_CHANNEL0;
extern const int SEG_OUT0_CHANNEL1;

extern const int SEG_OUT1_CHANNEL0;
extern const int SEG_OUT1_CHANNEL1;
extern const int SEG_OUT1_CHANNEL2;

extern const float CONF_THRESH;
extern const float NMS_THRESH;

extern const int WARMUP_TIME;

extern const int METER;
extern const int WATER;
extern const int LEVEL;

extern const std::vector<cv::Scalar> COLORS;

#endif