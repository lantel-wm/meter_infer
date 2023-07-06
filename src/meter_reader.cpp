#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "meter_reader.hpp"
#include "yolo.hpp"
#include "config.hpp"
#include "common.hpp"

void view_crops(std::vector<CropInfo> crops_meter, std::vector<CropInfo> crops_water)
{
    char crop_savepath[50];
    for (int i = 0; i < crops_meter.size(); i++)
    {
        sprintf(crop_savepath, "./crop/crop_meter_%d.png", i);
        cv::imwrite(crop_savepath, crops_meter[i].crop);
    }

    for (int i = 0; i < crops_water.size(); i++)
    {
        sprintf(crop_savepath, "./crop/crop_water_%d.png", i);
        cv::imwrite(crop_savepath, crops_water[i].crop);
    }
}

meterReader::meterReader(std::string const trt_model_det, std::string const trt_model_seg):
    detect(trt_model_det), segment(trt_model_seg)
{   
    LOG(INFO) << "loading detector";
    // detect = Detect(trt_model_det);
    detect.engineInfo();
    LOG(INFO) << "detector loaded";

    LOG(INFO) << "loading segmenter";
    // segment = Segment(trt_model_seg);
    segment.engineInfo();
    LOG(INFO) << "segmenter loaded";

    rect_scale = new uint8_t[RECT_WIDTH * RECT_HEIGHT]; // 360 * 40
    rect_pointer = new uint8_t[RECT_WIDTH * RECT_HEIGHT]; // 360 * 40
    line_scale = new int[RECT_WIDTH]; // 512
    line_pointer = new int[RECT_WIDTH]; // 512

    CUDA_CHECK(cudaMalloc((void**)&d_rect_scale, RECT_WIDTH * RECT_HEIGHT * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_rect_pointer, RECT_WIDTH * RECT_HEIGHT * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_circle_scale, CIRCLE_WIDTH * CIRCLE_HEIGHT * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_circle_pointer, CIRCLE_WIDTH * CIRCLE_HEIGHT * sizeof(uint8_t)));

}

meterReader::~meterReader()
{
    delete[] rect_scale;
    delete[] rect_pointer;
    delete[] line_scale;
    delete[] line_pointer;
    CUDA_CHECK(cudaFree(d_rect_scale));
    CUDA_CHECK(cudaFree(d_rect_pointer));
    CUDA_CHECK(cudaFree(d_circle_scale));
    CUDA_CHECK(cudaFree(d_circle_pointer));
}

void meterReader::read(std::vector<FrameInfo> &frame_batch, std::vector<MeterInfo> &meters)
{
    auto t1 = clock();
    crop_meters(frame_batch);
    auto t2 = clock();
    LOG(WARNING) << "crop_meters time: " << (t2 - t1) / 1000.0 << "ms";

    read_number(meters);

    draw_boxes(frame_batch);
}

void meterReader::crop_meters(std::vector<FrameInfo> &frame_batch)
{
    int batch_size = frame_batch.size();

    auto t1 = clock();
    detect.detect(frame_batch);
    auto t2 = clock();
    LOG(WARNING) << "detection time: " << (t2 - t1) / 1000.0 << "ms";

    crops_meter.clear();
    crops_water.clear();

    for (auto &frame_info : frame_batch)
    {
        for (auto &obj : frame_info.det_objs)
        {
            CropInfo crop_info;
            cv::Mat crop = frame_info.frame(obj.rect);
            crop_info.crop = crop;
            crop_info.class_id = obj.class_id;
            crop_info.rect = obj.rect;
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
    // view_crops(crops_meter, crops_water);

    // break the crops into batches of 8
    for (int i = 0; i < crops_meter.size(); i += 8)
    {
        std::vector<CropInfo>::const_iterator first = crops_meter.begin() + i;
        std::vector<CropInfo>::const_iterator last = MIN(crops_meter.begin() + i + 8, crops_meter.end());
        std::vector<CropInfo> crops_meter_batch;
        crops_meter_batch.assign(first, last);
        int batch_size = crops_meter_batch.size();
        LOG(INFO) << "segmenting " << crops_meter_batch.size() << " crops";
        t1 = clock();
        segment.segment(crops_meter_batch);
        t2 = clock();
        LOG(WARNING) << "segmentation time: " << (t2 - t1) / 1000.0 << "ms";
        for (int j = 0; j < batch_size; j++)
        {
            crops_meter[i + j].mask_pointer = crops_meter_batch[j].mask_pointer;
            crops_meter[i + j].mask_scale = crops_meter_batch[j].mask_scale;
        }
    }

    // cv::imwrite("mask_pointer.png", crops_meter[0].mask_pointer);
    // cv::imwrite("mask_scale.png", crops_meter[0].mask_scale);

    // LOG_ASSERT(0) << " stop here";
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