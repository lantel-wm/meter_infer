#ifndef METER_READER_HPP
#define METER_READER_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "config.hpp"

struct Meter
{
    std::string name; // name of the meter. e.g. pressure, water_level, etc.
    std::string unit; // unit of the reading. e.g. kPa, percent, etc.
    int id; // id of the meter
    float value; // value of the reading
    cv::Mat image; // image of the meter    
    cv::Point upper_left; // upperleft point of the meter
    cv::Point lower_right; // lowerright point of the meter
};


class meterReader
/*
    * This class is used to read the meter in the image.
    * It will read the image from the source and return the readings of the meters.
    * The source can be a image, a video or a rstp stream.
    * The readings will be stored in a vector of Reading.
    * The meters will be stored in a vector of Meter.
*/

{
    private:
        cv::Mat image; // current frame of the camera
        std::vector<Meter> meters; // meters in the frame

    public:
        meterReader(cv::Mat &image);
        ~meterReader();

        void crop_meters(); // input this->image, output this->meters
        void read_meters(); // input this->meters, output this->readings

        std::vector<Meter> get_meters(); // return all the meters
        Meter get_meter(int id); // return the meter with the id
        Meter get_meter(std::string name); // return the meter with the name

};

#endif