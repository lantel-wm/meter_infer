#include <ctime>
#include <iostream>
#include <time.h>
#include <thread>
#include <mutex>
#include <queue>

#include "common.hpp"
#include "cmdline.hpp"
#include "pro_con.hpp"
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

    std::vector<std::string> stream_urls = {
        "rtsp://admin:a120070001@192.168.1.100:554/Streaming/Channels/201",
        // "rtsp://admin:a120070001@192.168.1.100:554/Streaming/Channels/201",
        // "rtsp://admin:a120070001@192.168.1.100:554/Streaming/Channels/201",
        // "rtsp://admin:a120070001@192.168.1.100:554/Streaming/Channels/201",
        // "rtsp://admin:a120070001@192.168.1.100:554/Streaming/Channels/201",
        // "rtsp://admin:a120070001@192.168.1.100:554/Streaming/Channels/201",
        // "rtsp://admin:a120070001@192.168.1.100:554/Streaming/Channels/201",
        // "rtsp://admin:a120070001@192.168.1.100:554/Streaming/Channels/201",
        // "/home/zzy/cublas_test/data/201.mp4", 
        // "/home/zzy/cublas_test/data/201.mp4",
        // "/home/zzy/cublas_test/data/201.mp4", 
        // "/home/zzy/cublas_test/data/201.mp4",
        // "/home/zzy/cublas_test/data/201.mp4", 
        // "/home/zzy/cublas_test/data/201.mp4",
        // "/home/zzy/cublas_test/data/201.mp4", 
        // "/home/zzy/cublas_test/data/201.mp4",
    };

    if (stream_urls.size() != num_cam)
    {
        LOG(ERROR) << "number of stream urls not equal to number of cameras";
        exit(1);
    }

    // init meter reader
    std::string det_model = "yolov8n_batch" + std::to_string(det_batch) + ".trt";
    std::string seg_model = "yolov8s-seg_batch" + std::to_string(seg_batch) + ".trt";
    
    run(num_cam, num_cam * 4, stream_urls, det_batch, seg_batch, det_model, seg_model);
    
    return 0;
}