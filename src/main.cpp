#include <ctime>
#include <iostream>
#include <time.h>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

#include "common.hpp"
#include "stream.hpp"
#include "meter_reader.hpp"
#include "yolo.hpp"
#include "config.hpp"

using namespace google;

int main(int argc, char **argv)
{
    // cv::Mat img = cv::imread(IMAGE_PATH + "23.jpg");
    // cv::imshow("test", img);
    // cv::waitKey(0);
    // exit(0);
    
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::WARNING;
    FLAGS_log_dir = LOG_PATH;

    LOG(INFO) << "program started";

    // init meter reader
    meterReader meter_reader("yolov8n_batch8.trt", "yolov8s-seg_batch8.trt");
    
    Stream stream(IMAGE_PATH + "60.png");
    // Stream stream(IMAGE_PATH + "23.jpg");
    // Stream stream(VIDEO_PATH + "201.mp4");

    std::vector<MeterInfo> meters;

    // get frame batch
    while (stream.is_open())
    {
        std::vector<FrameInfo> frame_batch; // stores 8 frames
        // std::vector<std::vector<DetObject> > det_objs_batch(8); // stores 8 det results
        cv::Mat frame;

        stream.get_frame(frame);
        if (frame.empty())
        {
            LOG(WARNING) << "empty frame";
            continue;
        }

        for (int i = 0; i < 8; i++)
        {
            FrameInfo frame_info;
            frame_info.frame = frame;
            frame_batch.push_back(frame_info);        
        }

        auto t1 = clock();
        meter_reader.read(frame_batch, meters);
        auto t2 = clock();
        LOG(WARNING) << "read time: " << (t2 - t1) / 1000.0 << "ms";
        LOG(WARNING) << "------------------------------------------------------";
    }
    return 0;
}