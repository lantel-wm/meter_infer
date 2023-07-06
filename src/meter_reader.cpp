#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "meter_reader.hpp"
#include "yolo.hpp"
#include "config.hpp"

meterReader::meterReader(std::string const trt_model_det, std::string const trt_model_seg)
{   
    LOG(INFO) << "loading detector";
    detect = Detect(trt_model_det);
    detect.engineInfo();
    LOG(INFO) << "detector loaded";

    LOG(INFO) << "loading segmenter";
    segment = Segment(trt_model_seg);
    segment.engineInfo();
    LOG(INFO) << "segmenter loaded";

}

meterReader::~meterReader()
{
    ~detect();
    ~segment();
}

void meterReader::read(std::vector<FrameInfo> &frame_batch)
{
    crop_meters(frame_batch);
    draw_boxes(frame_batch);
}

void meterReader::crop_meters(std::vector<FrameInfo> &frame_batch)
{
    int batch_size = frame_batch.size();

    auto t1 = clock();
    detect.detect(frame_batch);
    auto t2 = clock();
    LOG(WARNING) << "detection time: " << (t2 - t1) / 1000.0 << "ms";

    crops.clear();

    for (auto &frame_info : frame_batch)
    {
        for (auto &obj : frame_info.det_objs)
        {
            CropInfo crop_info;
            cv::Mat crop = frame_info.frame(obj.rect);
            crop_info.crop = crop;
            crop_info.class_id = obj.class_id;
            if (crop_info.class_id == 0) // meter
            {
                crops_meter.push_back(crop_info);
            }
            else if (crop_info.class_id == 1) // water
            {
                crops_water.push_back(crop_info);
            }
        }
    }

    segment.segment(crops_meter);

    cv::imwrite("mask_pointer.png", crops_meter[0].mask_pointer);
    cv::imwrite("mask_scale.png", crops_meter[0].mask_scale);

    LOG_ASSERT(0) << " stop here";
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