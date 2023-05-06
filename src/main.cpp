#include <ctime>
#include <iostream>
#include <opencv2/flann/defines.h>
#include <opencv2/imgcodecs.hpp>
#include <time.h>
#include <vector>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <queue>
#include <condition_variable>

#include <opencv2/opencv.hpp>

#include "common.hpp"
#include "glog/logging.h"
#include "stream_to_img.hpp"
// #include "meter_reader.hpp"
#include "detect.hpp"
#include "config.hpp"

using namespace google;

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::WARNING;
    FLAGS_log_dir = LOG_PATH;
    LOG(INFO) << "program started";

    LOG(INFO) << "loading engine";
    Detect detect("yolov8n_batch8.trt");
    detect.engineInfo();
    LOG(INFO) << "engine loaded";

    std::vector<DetObject> det_objs;

    stream_to_img stream(IMAGE_PATH + "60.png");
    // stream_to_img stream(VIDEO_PATH + "201.mp4");
    cv::Mat frame;

    while (stream.is_open())
    {
        stream.get_frame(frame);
        if (frame.empty())
        {
            LOG(WARNING) << "empty frame";
            continue;
        }
        LOG(INFO) << "frame size: " << frame.size();
        auto t1 = clock();
        detect.detect(frame, det_objs);
        auto t2 = clock();
        LOG(WARNING) << "detection time: " << (t2 - t1) / 1000.0 << "ms";

        DUMP_OBJ_INFO(det_objs);
        for (auto &obj : det_objs)
        {
            cv::Mat crop = frame(obj.rect);
            cv::imwrite("crop" + std::to_string(obj.class_id) + ".png", crop);    
            cv::rectangle(frame, obj.rect, cv::Scalar(0, 0, 255), 2);
            cv::putText(frame, obj.name, cv::Point(obj.rect.x, obj.rect.y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        }
        cv::imwrite("result.png", frame);
    }
    return 0;
}