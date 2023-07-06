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
        void read_number(); // input this->meters, output this->readings
        void draw_boxes(std::vector<FrameInfo> &images); // draw the bounding box of the detected objects

    public:
        meterReader(std::string const trt_model_det, std::string const trt_model_seg);
        ~meterReader();

        void read(std::vector<FrameInfo> &frame_batch);
        
};

#endif