#include <ctime>
#include <iostream>
#include <fstream>
#include <time.h>
#include <thread>
#include <mutex>
#include <queue>

#include "common.hpp"
#include "cmdline.hpp"
#include "pro_con.hpp"
#include "meter_reader.hpp"
#include "yolo.hpp"
#include "json.hpp"
#include "config.hpp"

using namespace google;
using json = nlohmann::json;

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
    parser.add<int>("debug", 'b', "debug mode", false, 0, cmdline::range(0, 1));
    parser.parse_check(argc, argv);

    LOG(INFO) << "num_cam: " << parser.get<int>("num_cam");
    LOG(INFO) << "det_batch: " << parser.get<int>("det_batch");
    LOG(INFO) << "seg_batch: " << parser.get<int>("seg_batch");
    LOG(INFO) << "source: " << parser.get<std::string>("source");

    int num_cam = parser.get<int>("num_cam");
    int det_batch = parser.get<int>("det_batch");
    int seg_batch = parser.get<int>("seg_batch");
    std::string source = parser.get<std::string>("source");
    int debug_on = parser.get<int>("debug"); // 1: debug mode, 0: normal mode

    std::fstream fs(PROJECT_PATH + "config.json");
    json data = json::parse(fs);

    std::vector<std::string> stream_urls = data["stream_urls"];
    std::string det_model_format = data["det_model"];
    std::string seg_model_format = data["seg_model"];
    char det_model_cstr[100];
    char seg_model_cstr[100];
    sprintf(det_model_cstr, det_model_format.c_str(), det_batch);
    sprintf(seg_model_cstr, seg_model_format.c_str(), seg_batch);
    std::string det_model(det_model_cstr);
    std::string seg_model(seg_model_cstr);

    LOG(WARNING) << "det_model: " << det_model;
    LOG(WARNING) << "seg_model: " << seg_model;

    if (stream_urls.size() != num_cam)
    {
        LOG(ERROR) << "number of stream urls not equal to number of cameras";
        exit(1);
    }

    // model file not found
    // if (access(det_model.c_str(), F_OK) == -1)
    // {
    //     LOG(ERROR) << "detector model file not found, check \'det_model\' in config.json";
    //     exit(1);
    // }
    // if (!access(seg_model.c_str(), F_OK) == -1)
    // {
    //     LOG(ERROR) << "segmenter model file not found, check \'seg_model\' in config.json";
    //     exit(1);
    // }
    
    run(num_cam, num_cam * 4, stream_urls, det_batch, seg_batch, det_model, seg_model, debug_on);
    
    return 0;
}