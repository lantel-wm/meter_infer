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
        std::vector<CropInfo> crops_meter; // cropped meter
        std::vector<CropInfo> crops_water; // cropped water

        void crop_meters(std::vector<FrameInfo> &frame_batch);
        void read_number(std::vector<MeterInfo> &meters);
        void draw_boxes(std::vector<FrameInfo> &images); // draw the bounding box of the detected objects

        void read_meter(std::vector<CropInfo> &crops_meter, std::vector<MeterInfo> &meters);
        void minimum_coverage_circle(std::vector<cv::Point> &points, int &radius, cv::Point &center);

        uint8_t* rect_scale;
        uint8_t* rect_pointer;
        uint8_t* d_rect_scale; // device pointer
        uint8_t* d_rect_pointer; // device pointer
        uint8_t* d_circle_scale; // device pointer
        uint8_t* d_circle_pointer; // device pointer

    public:
        meterReader(std::string const trt_model_det, std::string const trt_model_seg);
        ~meterReader();

        void read(std::vector<FrameInfo> &frame_batch, std::vector<MeterInfo> &meters);
        
};

#endif