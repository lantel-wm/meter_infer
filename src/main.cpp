#include <ctime>
#include <iostream>
#include <time.h>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

#include "common.hpp"
#include "stream_to_img.hpp"
#include "meter_reader.hpp"
#include "yolo.hpp"
#include "config.hpp"

using namespace google;

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::WARNING;
    FLAGS_log_dir = LOG_PATH;

    LOG(INFO) << "program started";

    // std::vector<CropInfo> crops;
    // Segment segment("yolov8s-seg_batch8.trt");


    meterReader reader("yolov8n_batch8.trt", "233");

    stream_to_img stream(IMAGE_PATH + "60.png");
    // stream_to_img stream(VIDEO_PATH + "201.mp4");
    cv::Mat frame;

    while (stream.is_open())
    {
        std::vector<FrameInfo> frames; // stores 8 frames
        // std::vector<std::vector<DetObject> > det_objs_batch(8); // stores 8 det results

        stream.get_frame(frame);
        if (frame.empty())
        {
            LOG(WARNING) << "empty frame";
            continue;
        }

        for (int i = 0; i < 1; i++)
        {
            FrameInfo frame_info;
            frame_info.frame = frame;
            frames.push_back(frame_info);        
        }

        reader.read(frames);

    }
    return 0;
}