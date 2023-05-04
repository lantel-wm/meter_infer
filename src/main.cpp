#include <iostream>
#include <opencv2/flann/defines.h>
#include <vector>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <queue>
#include <condition_variable>

#include <opencv2/opencv.hpp>

#include "glog/logging.h"
#include "stream_to_img.hpp"
// #include "meter_reader.hpp"
#include "detect.hpp"
#include "config.hpp"

using namespace google;

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_log_dir = LOG_PATH;
    
    LOG(INFO) << "program started";
    
    stream_to_img stream(IMAGE_PATH + "60.png");
    cv::Mat frame;
    stream.get_frame(frame);
    // cv::imshow("frame", frame);
    // cv::waitKey(0);

    LOG(INFO) << "frame size: " << frame.size();

    std::vector<detObject> results;
    LOG(INFO) << "loading engine";
    Detect detect("yolov8n_batch8.trt");
    LOG(INFO) << "engine loaded";
    detect.detect(frame, results);

    return 0;
}