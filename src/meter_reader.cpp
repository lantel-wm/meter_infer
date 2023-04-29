#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "meter_reader.hpp"
#include "config.hpp"

meterReader::meterReader(cv::Mat &image)
{
    this->image = image;
}

meterReader::~meterReader()
{
}

void meterReader::crop_meters()
{
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