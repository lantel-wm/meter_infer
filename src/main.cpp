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
#include "meter_reader.hpp"
#include "detect.hpp"
#include "config.hpp"

using namespace google;

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::WARNING;
    FLAGS_log_dir = LOG_PATH;

    LOG(INFO) << "program started";

    meterReader reader("yolov8n_batch8.trt", "233");

    stream_to_img stream(IMAGE_PATH + "60.png");
    // stream_to_img stream(VIDEO_PATH + "201.mp4");
    cv::Mat frame;

    while (stream.is_open())
    {
        std::vector<frameInfo> frames; // stores 8 frames
        // std::vector<std::vector<DetObject> > det_objs_batch(8); // stores 8 det results

        stream.get_frame(frame);
        if (frame.empty())
        {
            LOG(WARNING) << "empty frame";
            continue;
        }

        for (int i = 0; i < 8; i++)
        {
            frameInfo frame_info;
            frame_info.frame = frame;
            frames.push_back(frame_info);        
        }

        reader.read(frames);

    }
    return 0;
}