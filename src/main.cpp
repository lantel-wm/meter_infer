#include <ctime>
#include <iostream>
#include <time.h>
#include <thread>
#include <mutex>
#include <queue>

#include "common.hpp"
#include "cmdline.hpp"
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
    
    FLAGS_stderrthreshold = WARNING;
    FLAGS_log_dir = LOG_PATH;
    // FLAGS_minloglevel = WARNING;
    FLAGS_logbuflevel = ERROR;
    InitGoogleLogging(argv[0]);

    
    LOG(INFO) << "program started";

    // sleep(10);
    // exit(0);
    

    cmdline::parser parser;
    parser.add<int>("num_cam", 'n', "number of cameras", false, 8, cmdline::range(1, 8));
    parser.add<int>("det_batch", 'd', "batch size of detector", false, 8, cmdline::oneof<int>(1, 2, 4, 8));
    parser.add<int>("seg_batch", 'g', "batch size of segmenter", false, 8, cmdline::oneof<int>(1, 2, 4, 8));
    parser.add<std::string>("source", 's', "path to video or image, or rstp address", false, VIDEO_PATH + "201.mp4");
    parser.parse_check(argc, argv);

    LOG(INFO) << "num_cam: " << parser.get<int>("num_cam");
    LOG(INFO) << "det_batch: " << parser.get<int>("det_batch");
    LOG(INFO) << "seg_batch: " << parser.get<int>("seg_batch");
    LOG(INFO) << "source: " << parser.get<std::string>("source");

    int num_cam = parser.get<int>("num_cam");
    int det_batch = parser.get<int>("det_batch");
    int seg_batch = parser.get<int>("seg_batch");
    std::string source = parser.get<std::string>("source");

    // init meter reader
    std::string det_model = "yolov8n_batch" + std::to_string(det_batch) + ".trt";
    std::string seg_model = "yolov8s-seg_batch" + std::to_string(seg_batch) + ".trt";
    meterReader meter_reader(det_model, seg_model);
    
    // Stream stream(IMAGE_PATH + "60.png");
    // Stream stream(IMAGE_PATH + "23.jpg");
    Stream stream(VIDEO_PATH + "201.mp4");

    std::vector<MeterInfo> meters;

    // get frame batch
    while (stream.is_open())
    {
        std::vector<FrameInfo> frame_batch; // stores num_cam frames
        cv::Mat frame;

        stream.get_frame(frame);
        if (frame.empty())
        {
            LOG(WARNING) << "empty frame";
            continue;
        }

        for (int i = 0; i < num_cam; i++)
        {
            FrameInfo frame_info;
            frame_info.frame = frame;
            frame_batch.push_back(frame_info);        
        }

        auto t1 = clock();
        meter_reader.read(frame_batch, meters);
        auto t2 = clock();
        // printf("read time: %fms\n", (t2 - t1) / 1000.0);

        LOG(INFO) << "meters: " << meters.size();
        for (auto &meter : meters)
        {
            meter.dump();
        }

        LOG(WARNING) << "read time: " << (t2 - t1) / 1000.0 << "ms";
    }
    return 0;
}