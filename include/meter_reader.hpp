#ifndef _METER_READER_HPP_
#define _METER_READER_HPP_

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "config.hpp"
#include "yolo.hpp"

class meterReader
{
    private:
        Detect detect; // detector
        Segment segment; // segmenter

        // TODO: use a 2d vector to store different kinds of meters
        std::vector<CropInfo> crops_meter; // cropped meter
        std::vector<CropInfo> crops_water; // cropped water

        void crop_meters(std::vector<FrameInfo> &frame_batch);
        void parse_meters(); // recognize the scales and pointers of the meters
        void read_number(std::vector<MeterInfo> &meters);
        // void draw_boxes(std::vector<FrameInfo> &frame_batch, std::vector<MeterInfo> meters); // draw the bounding box of the detected objects

        void read_meter(std::vector<CropInfo> &crops_meter, std::vector<MeterInfo> &meters);
        void minimum_coverage_circle(std::vector<cv::Point2f> points, int &radius, cv::Point &center);

        void read_water(std::vector<CropInfo> &crops_water, std::vector<MeterInfo> &meters);

    public:
        meterReader(std::string const trt_model_det, std::string const trt_model_seg);
        ~meterReader();

        bool read(std::vector<FrameInfo> &frame_batch, std::vector<MeterInfo> &meters);
        
};

#endif