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

    std::vector<CropInfo> crops;
    Segment segment("yolov8s-seg_batch8.trt");
    segment.engineInfo();
    cv::Mat meter_img = cv::imread(IMAGE_PATH + "meter.png");
    cv::resize(meter_img, meter_img, cv::Size(640, 640));
    for (int i = 0; i < 8; i++)
    {
        CropInfo crop_info;
        crop_info.crop = meter_img;
        crops.push_back(crop_info);
    }
    segment.segment(crops);
    for(int i = 0; i < 1; i++)
    {
        cv::Mat det = crops[i].crop;
        std::vector<DetObject> det_objs = crops[i].det_objs;
        for (auto &det_obj : det_objs)
        {
            cv::rectangle(det, det_obj.rect, cv::Scalar(0, 0, 255), 1);
        }
        cv::imwrite("seg_det.png", det);
    }
    // LOG_ASSERT(0) << " stop here";

    Segment::processMask(crops);

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

        for (int i = 0; i < 8; i++)
        {
            FrameInfo frame_info;
            frame_info.frame = frame;
            frames.push_back(frame_info);        
        }

        reader.read(frames);

    }
    return 0;
}