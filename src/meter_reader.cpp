#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "common.hpp"
#include "meter_reader.hpp"
#include "yolo.hpp"
#include "config.hpp"


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

// load the detection model and segmentation model
meterReader::meterReader(std::string const trt_model_det, std::string const trt_model_seg, int det_batch, int seg_batch):
    detect(trt_model_det), segment(trt_model_seg)
{   
    det_batch = det_batch;
    seg_batch = seg_batch;

    LOG(INFO) << "loading detector";
    // detect = Detect(trt_model_det);
    detect.engineInfo();
    LOG(INFO) << "detector loaded";

    LOG(INFO) << "loading segmenter";
    // segment = Segment(trt_model_seg);
    segment.engineInfo();
    LOG(INFO) << "segmenter loaded";

    // rect_scale = new uint8_t[RECT_WIDTH * RECT_HEIGHT]; // 360 * 40
    // rect_pointer = new uint8_t[RECT_WIDTH * RECT_HEIGHT]; // 360 * 40
    // line_scale = new int[RECT_WIDTH]; // 512
    // line_pointer = new int[RECT_WIDTH]; // 512

    // CUDA_CHECK(cudaMalloc((void**)&d_rect_scale, RECT_WIDTH * RECT_HEIGHT * sizeof(uint8_t)));
    // CUDA_CHECK(cudaMalloc((void**)&d_rect_pointer, RECT_WIDTH * RECT_HEIGHT * sizeof(uint8_t)));
    // CUDA_CHECK(cudaMalloc((void**)&d_circle_scale, CIRCLE_WIDTH * CIRCLE_HEIGHT * sizeof(uint8_t)));
    // CUDA_CHECK(cudaMalloc((void**)&d_circle_pointer, CIRCLE_WIDTH * CIRCLE_HEIGHT * sizeof(uint8_t)));

}

meterReader::~meterReader()
{
    // delete[] rect_scale;
    // delete[] rect_pointer;
    // delete[] line_scale;
    // delete[] line_pointer;
    // CUDA_CHECK(cudaFree(d_rect_scale));
    // CUDA_CHECK(cudaFree(d_rect_pointer));
    // CUDA_CHECK(cudaFree(d_circle_scale));
    // CUDA_CHECK(cudaFree(d_circle_pointer));
}

// recognize the instruments in the frames and sort them by y coordinate then x coordinate
void meterReader::recognize(std::vector<FrameInfo> &frame_batch)
{
    // TODO: use a 2d vector to store different kinds of meters
    detect.detect(frame_batch);

    for (int ibatch = 0; ibatch < frame_batch.size(); ibatch++)
    {
        std::vector<DetObject> objs = frame_batch[ibatch].det_objs;
        // sort the objects by y coordinate then x coordinate
        std::sort(objs.begin(), objs.end(), 
            [](DetObject a, DetObject b) { 
                return a.rect.y == b.rect.y? a.rect.x < b.rect.x: a.rect.y < b.rect.y; 
            }
        );
        frame_batch[ibatch].det_objs = objs;
    }
}

void meterReader::set_camera_instrument_id(std::vector<FrameInfo> frame_batch)
{
    camera_instrument_id.clear();
    int num_cam = frame_batch.size();
    for (int camera_id = 0; camera_id < num_cam; camera_id++)
    {
        camera_instrument_id.push_back(frame_batch[camera_id].det_objs);
    }
}

int meterReader::get_instrument_num()
{
    int num = 0;
    for (int i = 0; i < camera_instrument_id.size(); i++)
    {
        num += camera_instrument_id[i].size();
    }
    return num;
}

bool meterReader::read_error(std::vector<FrameInfo> &frame_batch)
{
    for (int ibatch = 0; ibatch < frame_batch.size(); ibatch++)
    {
        std::vector<DetObject> objs = frame_batch[ibatch].det_objs;
        int camera_id = frame_batch[ibatch].camera_id;
        if (objs.size() != camera_instrument_id[camera_id].size())
        {
            return true;
        }
        for (int iobj = 0; iobj < objs.size(); iobj++)
        {
            if (objs[iobj].class_id != camera_instrument_id[camera_id][iobj].class_id)
            {
                return true;
            }
        }
    }
    return false;
}

// if read error, return true
bool meterReader::read(std::vector<FrameInfo> &frame_batch, std::vector<MeterInfo> &meters)
{
    // TODO: use a 2d vector to store different kinds of meters
    std::vector<CropInfo> crops_meter; // cropped meter
    std::vector<CropInfo> crops_water; // cropped water
    
    meters.clear();
    // auto t1 = clock();
    crop_meters(frame_batch, crops_meter, crops_water);
    // auto t2 = clock();
    // LOG(WARNING) << "crop_meters time: " << (t2 - t1) / 1000.0 << "ms";



    if (crops_meter.size() == 0 && crops_water.size() == 0)
    {
        // LOG(WARNING) << "No meter detected";
        meters.clear();
        return true;
    }

    // if (read_error(frame_batch))
    // {
    //     // LOG(WARNING) << "Read error";
    //     meters.clear();
    //     return true;
    // }

    parse_meters(crops_meter, crops_water);

    read_number(meters, crops_meter, crops_water);

    return false;
    // draw_boxes(frame_batch, meters);
}

// run object detection on the frames to get meter crops
void meterReader::crop_meters(std::vector<FrameInfo> &frame_batch, std::vector<CropInfo> &crops_meter, std::vector<CropInfo> &crops_water)
{
    int batch_size = frame_batch.size();
    LOG(INFO) << "detecting " << batch_size << " frames";

    // auto t1 = std::chrono::high_resolution_clock::now();
    detect.detect(frame_batch);
    // auto t2 = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    // LOG(WARNING) << "detection time: " << duration << "ms";
    
    // // TODO: used a 2d vector to store more kinds of meters
    // crops_meter.clear();
    // crops_water.clear();

    for (int ibatch = 0; ibatch < batch_size; ibatch++)
    {
        FrameInfo frame_info = frame_batch[ibatch];
        std::vector<DetObject> objs = frame_info.det_objs;

        std::sort(objs.begin(), objs.end(), 
            [](DetObject a, DetObject b) { 
                return a.rect.y == b.rect.y? a.rect.x < b.rect.x: a.rect.y < b.rect.y; 
            }
        );

        // set instrument_id by camera_instrument_id
        for (int iobj = 0; iobj < objs.size(); iobj++)
        {
            int camera_id = frame_info.camera_id;
            objs[iobj].instrument_id = camera_instrument_id[camera_id][iobj].instrument_id;
        }

        for (auto &obj : objs)
        {
            CropInfo crop_info;
            frame_info.frame(obj.rect).copyTo(crop_info.crop);
            crop_info.class_id = obj.class_id;
            crop_info.frame_batch_id = ibatch;
            crop_info.camera_id = frame_info.camera_id;
            crop_info.instrument_id = obj.instrument_id;
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
    // LOG_ASSERT(0) << " stop here";
}

// parse the meter crops to get information to read the meter
void meterReader::parse_meters(std::vector<CropInfo> &crops_meter, std::vector<CropInfo> &crops_water)
{
    // meter segmentation
    // break the crops into batches of seg_batch
    for (int i = 0; i < crops_meter.size(); i += seg_batch)
    {
        std::vector<CropInfo>::const_iterator first = crops_meter.begin() + i;
        std::vector<CropInfo>::const_iterator last = MIN(crops_meter.begin() + i + seg_batch, crops_meter.end());
        std::vector<CropInfo> crops_meter_batch;
        crops_meter_batch.assign(first, last);
        int batch_size = crops_meter_batch.size();
        LOG(INFO) << "segmenting " << crops_meter_batch.size() << " crops";
        // auto t1 = std::chrono::high_resolution_clock::now();
        segment.segment(crops_meter_batch);
        // auto t2 = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        // LOG(WARNING) << "segmentation time: " << duration << "ms";
        
        for (int j = 0; j < batch_size; j++)
        {
            crops_meter[i + j].mask_pointer = crops_meter_batch[j].mask_pointer;
            crops_meter[i + j].mask_scale = crops_meter_batch[j].mask_scale;
        }
    }

    // water detection
    // break the crops into batches of det_batch
    for (int i = 0; i < crops_water.size(); i += det_batch)
    {
        std::vector<FrameInfo> crops_water_batch;
        int first = i;
        int last = MIN(i + det_batch, crops_water.size());
        int batch_size = last - first;

        for (int j = first; j < last; j++)
        {
            FrameInfo frame_info;
            frame_info.frame = crops_water[j].crop;
            frame_info.info = "water crops";
            crops_water_batch.push_back(frame_info);
        }
        LOG(INFO) << "detecting " << crops_water_batch.size() << " crops";
        // t1 = clock();
        detect.detect(crops_water_batch);
        // t2 = clock();
        // LOG(WARNING) << "detection time: " << (t2 - t1) / 1000.0 << "ms";
        for (int j = 0; j < batch_size; j++)
        {
            crops_water[i + j].det_objs = crops_water_batch[j].det_objs;
        }

        // cv::Mat water_det = crops_water[i].crop.clone();
        // for (auto &obj : crops_water[i].det_objs)
        // {
        //     cv::rectangle(water_det, obj.rect, cv::Scalar(0, 0, 255), 2);
        // }
        // cv::imwrite("water_det.png", water_det);
        // LOG_ASSERT(0) << " stop here";
    }

    // cv::imwrite("mask_pointer.png", crops_meter[0].mask_pointer);
    // cv::imwrite("mask_scale.png", crops_meter[0].mask_scale);

}

// void meterReader::draw_boxes(std::vector<FrameInfo> &frame_batch, std::vector<MeterInfo> meters)
// {
//     // for (auto &image : images)
//     // {
//     //     for (auto &obj : image.det_objs)
//     //     {
//     //         std::string display_text = obj.class_name + " " + obj.meter_reading;
//     //         cv::Scalar color = COLORS[obj.class_id];
//     //         cv::rectangle(image.frame, obj.rect, color, 2);
//     //         cv::putText(image.frame, display_text, cv::Point(obj.rect.x, obj.rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
//     //     }
//     //     // cv::imwrite("result.png", image.frame);
//     // }
    
//     for (int ibatch = 0; ibatch < frame_batch.size(); ibatch++)
//     {
//         FrameInfo frame_info = frame_batch[ibatch];
//         int img_width = frame_info.frame.cols;
//         int img_height = frame_info.frame.rows;

//         for (auto &meter_info: meters)
//         {
//             if (meter_info.frame_batch_id != ibatch)
//                 continue;
            
//             std::string display_text = meter_info.class_name + " " + meter_info.meter_reading;
//             cv::Scalar color = COLORS[meter_info.class_id];
//             cv::rectangle(frame_info.frame, meter_info.rect, color, 2);
//             cv::putText(frame_info.frame, display_text, cv::Point(meter_info.rect.x, meter_info.rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
//         }
//     }
//     // cv::imwrite("result.png", frame_batch[0].frame);
//     // sleep(1);
//     // cv::imshow("result", frame_batch[0].frame);
// }