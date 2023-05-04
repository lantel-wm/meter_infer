#include <iostream>
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

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_log_dir = LOG_PATH;

    stream_to_img stream(IMAGE_PATH + "60.png");
    cv::Mat frame;
    stream.get_frame(frame);
    // cv::imshow("frame", frame);
    // cv::waitKey(0);


    std::vector<detectResult> results;
    Detect detect("yolov8n_batch8.trt");
    detect.Infer(frame, results);

    for (auto &result : results)
    {
        std::cout << result.name << " " << result.conf << std::endl;
        cv::rectangle(frame, result.rect, cv::Scalar(0, 0, 255), 2);
        cv::putText(frame, result.name, cv::Point(result.rect.x, result.rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    }

    cv::imshow("frame", frame);
    cv::waitKey(0);

    return 0;
}