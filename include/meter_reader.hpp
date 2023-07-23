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

        int det_batch; // batch size of detector
        int seg_batch; // batch size of segmenter

        // // TODO: use a 2d vector to store different kinds of meters
        // std::vector<CropInfo> crops_meter; // cropped meter
        // std::vector<CropInfo> crops_water; // cropped water

        void crop_meters(std::vector<FrameInfo> &frame_batch, std::vector<CropInfo> &crops_meter, std::vector<CropInfo> &crops_water);
        void parse_meters(std::vector<CropInfo> &crops_meter, std::vector<CropInfo> &crops_water); // recognize the scales and pointers of the meters
        void read_number(std::vector<MeterInfo> &meters, std::vector<CropInfo> &crops_meter, std::vector<CropInfo> &crops_water);
        // void draw_boxes(std::vector<FrameInfo> &frame_batch, std::vector<MeterInfo> meters); // draw the bounding box of the detected objects

        void read_meter(std::vector<CropInfo> &crops_meter, std::vector<MeterInfo> &meters);
        void minimum_coverage_circle(std::vector<cv::Point2f> points, int &radius, cv::Point &center);

        void read_water(std::vector<CropInfo> &crops_water, std::vector<MeterInfo> &meters);

    public:
        meterReader(std::string const trt_model_det, std::string const trt_model_seg, int det_batch, int seg_batch);
        ~meterReader();

        bool read(std::vector<FrameInfo> &frame_batch, std::vector<MeterInfo> &meters);

        int get_det_batch() { return det_batch; }
        int get_seg_batch() { return seg_batch; }
        
};

#endif