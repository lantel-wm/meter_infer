#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "meter_reader.hpp"
#include "yolo.hpp"
#include "config.hpp"

meterReader::meterReader(std::string const trt_model_det, std::string const trt_model_seg)
    : detect(trt_model_det) 
{   
    LOG(INFO) << "loading detector";
    this->detect = detect;
    this->detect.engineInfo();
    LOG(INFO) << "detector loaded";

    // LOG(INFO) << "loading segmenter";
    // this->segment = Segment(trt_model_seg);
    // this->segment.engineInfo();
    // LOG(INFO) << "segmenter loaded";

}

meterReader::~meterReader()
{
}

void meterReader::read(std::vector<FrameInfo> &frames)
{
    this->crop_meters(frames);
    this->draw_boxes(frames);
}

void meterReader::crop_meters(std::vector<FrameInfo> &frames)
{
    int batch_size = frames.size();

    auto t1 = clock();
    this->detect.detect(frames);
    auto t2 = clock();
    LOG(WARNING) << "detection time: " << (t2 - t1) / 1000.0 << "ms";

    for (auto &frame_info : frames)
    {
        for (auto &obj : frame_info.det_objs)
        {
            cv::Mat crop = frame_info.frame(obj.rect);
            cv::imwrite("crop" + std::to_string(obj.class_id) + ".png", crop);
            if (obj.class_id == 0)
            {
                obj.class_name = "meter";
                obj.meter_reading = "2.3kPa";
            }
            else if (obj.class_id == 1)
            {
                obj.class_name = "water";
                obj.meter_reading = "66%";
            }
        }
    }
}

void meterReader::draw_boxes(std::vector<FrameInfo> &images)
{
    for (auto &image : images)
    {
        for (auto &obj : image.det_objs)
        {
            std::string display_text = obj.class_name + " " + obj.meter_reading;
            cv::Scalar color = COLORS[obj.class_id];
            cv::rectangle(image.frame, obj.rect, color, 2);
            cv::putText(image.frame, display_text, cv::Point(obj.rect.x, obj.rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
        }
        cv::imwrite("result.png", image.frame);
    }
}