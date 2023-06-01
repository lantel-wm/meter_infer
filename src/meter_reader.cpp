#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "meter_reader.hpp"
#include "config.hpp"

meterReader::meterReader(std::string const trt_model_det, std::string const trt_model_seg)
{
    LOG(INFO) << "loading detector";
    this->detect = Detect(trt_model_det);
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

void meterReader::crop_meters(std::vector<frame_info> &frames, std::vector<std::vector<DetObject> > &det_objs)
{
    int batch_size = frames.size();
    for (int i = 0; i < batch_size; i++)
    {
        for (auto &obj : det_objs[i])
        {
            cv::Mat crop = frames[i].frame(obj.rect);
            if 
            cv::imwrite("crop" + std::to_string(obj.class_id) + ".png", crop);    
        }
    }
}

void meterReader::read_meters()
{
}

std::vector<Meter> meterReader::get_meters()
{
    return this->meters;
}

Meter meterReader::get_meter(int id)
{
    return this->meters[id];
}

Meter meterReader::get_meter(std::string name)
{
    for (auto meter : this->meters)
    {
        if (meter.name == name)
        {
            return meter;
        }
    }
    LOG(ERROR) << "No meter with name " << name << " found.";
    return Meter();
}