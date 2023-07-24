#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <random>
#include <math.h>
#include <iostream>

#include "common.hpp"
#include "meter_reader.hpp"
#include "yolo.hpp"
#include "config.hpp"

#define PI 3.1415926f

// return a float range in [0, 1]
float location_to_reading(std::vector<float> p_loc, std::vector<float> s_loc)
{
    int num_scales = s_loc.size();
    
    if (p_loc.size() == 0 || s_loc.size() == 0)
        return -1.0f;

    if (p_loc[0] < s_loc[0])
        return 0.0f;
    if (p_loc[0] > s_loc[num_scales - 1])
        return 1.0f;

    for (int i = 0; i < num_scales - 1; i++)
    {
        if (p_loc[0] >= s_loc[i] && p_loc[0] <= s_loc[i + 1])
        {
            return ((float)i + (p_loc[0] - s_loc[i]) / (s_loc[i + 1] - s_loc[i])) / (num_scales - 1);
        }
    }

    return -1.0f;
}

// find the local maximums of a line as the location of the scale or pointer
void line_to_location(int *line, std::vector<float> &location, int width)
{
    float index_buffer[width];
    int ib_cur = 0; // pointer to index_buffer
    bool ascending = true;
    // find all local maximums
    for (int i = 1; i < width - 1; i++)
    {
        if (line[i] == 0)
        {
            continue;
        }

        if (line[i - 1] > line[i])
        {
            ascending = false;
            continue;
        }

        if (line[i - 1] < line[i] && line[i] > line[i + 1]) // 4 6 5
        {
            location.push_back((float)i);
            continue;
        }

        if (line[i - 1] <= line[i] && line[i] <= line[i + 1])
        {
            ascending = true;
        }

        if (line[i - 1] < line[i] && line[i] == line[i + 1]) // 4 6 6
        {
            index_buffer[ib_cur++] = (float)i;
            continue;
        }

        if (ascending && line[i - 1] == line[i] && line[i] == line[i + 1])
        {
            index_buffer[ib_cur++] = (float)i;
            continue;
        }

        if (ascending && line[i - 1] == line[i] && line[i] > line[i + 1])
        {
            index_buffer[ib_cur++] = (float)i;
            float mean = 0;
            for (int j = 0; j < ib_cur; j++)
            {
                mean += index_buffer[j];
            }
            mean /= ib_cur;
            location.push_back(mean);
            ib_cur = 0;
        }
    }
}

// __global__ void rect_to_line(uint8_t* rect, int* line, int width, int height)
// {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;

//     if (x >= width)
//         return;

//     line[x] = 0;
//     for (int y = 0; y < height; y++)
//         line[x] += rect[y * width + x];
// }

void rect_to_line_cpu(uint8_t* rect, int* line, int width, int height)
{
    float mean = 0;
    for (int x = 0; x < width; x++)
    {
        line[x] = 0;
        for (int y = 0; y < height; y++)
            line[x] += rect[y * width + x];
        mean += line[x];
    }
    mean /= width;

    for (int x = 0; x < width; x++)
        line[x] = line[x] > mean ? line[x] - mean : 0;
}

__global__ void circle_to_rect(uint8_t* circle, uint8_t* rect, int radius, cv::Point center,
    int rect_width, int rect_height, int circle_width, int circle_height)
{
    int d_rho = blockIdx.x * blockDim.x + threadIdx.x;
    int theta = blockIdx.y * blockDim.y + threadIdx.y;

    if (d_rho >= rect_height || theta >= rect_width)
        return;

    int rho = d_rho + radius - rect_height;
    int x = round(center.x + rho * cos(theta * 2 * PI / rect_width + PI / 2));
    int y = round(center.y + rho * sin(theta * 2 * PI / rect_width + PI / 2));

    if (x < 0 || x >= circle_width || y < 0 || y >= circle_height)
    {
        rect[d_rho * rect_width + theta] = 0;
        return;
    }

    rect[d_rho * rect_width + theta] = circle[y * circle_width + x];
    
}

void circle_to_rect_cpu(uint8_t* circle, uint8_t* rect, int radius, cv::Point center,
    int rect_width, int rect_height, int circle_width, int circle_height)
{
    for (int d_rho = 0; d_rho < rect_height; d_rho++)
    {
        for (int theta = 0; theta < rect_width; theta++)
        {
            int rho = d_rho + radius - rect_height;
            int x = round(center.x + rho * cos(theta * 2 * PI / rect_width + PI / 2));
            int y = round(center.y + rho * sin(theta * 2 * PI / rect_width + PI / 2));
            
            // LOG(INFO) << "x: " << x << ", y: " << y << ", rho: " << rho << ", theta: " << theta;

            if (x < 0 || x >= circle_width || y < 0 || y >= circle_height)
            {
                rect[d_rho * rect_width + theta] = 0;
                continue;
            }

            rect[d_rho * rect_width + theta] = circle[y * circle_width + x];
        }
    }
}

// find the minimum coverage circle of a set of points
// random incremental algorithm, time complexity O(n)
void meterReader::minimum_coverage_circle(std::vector<cv::Point2f> points, int &radius, cv::Point &center)
{
    std::random_shuffle(points.begin(), points.end());
    int n = points.size();
    float radius_f = 0; // use float instead of int to avoid error in calculation
    cv::Point2f center_f;
    center_f = points[0];
    radius_f = 0;

    for (int i = 1; i < n; i++)
    {
        if (cv::norm(points[i] - center_f) <= radius_f) // points[i] is not in circle
            continue;
        
        center_f = points[i];
        radius_f = 0;
        for (int j = 0; j < i; j++)
        {
            if (cv::norm(points[j] - center_f) <= radius_f) // points[j] is not in circle
                continue;

            center_f = (points[i] + points[j]) / 2;
            radius_f = cv::norm(points[i] - points[j]) / 2;
            for (int k = 0; k < j; k++)
            {
                if (cv::norm(points[k] - center_f) <= radius_f) // points[k] is not in circle
                    continue;
                
                // calculate the center and radius of the circle passing through points i, j, k
                cv::Point2f a = points[i] - points[j]; // AB
                cv::Point2f b = points[i] - points[k]; // AC
                cv::Point2f p = (points[i] + points[j]) / 2; // midpoint of AB
                cv::Point2f q = (points[i] + points[k]) / 2; // midpoint of AC
                cv::Point2f v = cv::Point2f(-a.y, a.x); // vector perpendicular to AB
                cv::Point2f w = cv::Point2f(-b.y, b.x); // vector perpendicular to AC
                float t1 = w.cross(p - q) / v.cross(w); // t1 = (q - p) x w / v x w
                center_f = p + t1 * v; // center = p + t1 * v
                radius_f = cv::norm(points[i] - center_f);
            }
        }
    }

    radius = (int)radius_f;
    center = cv::Point((int)center_f.x, (int)center_f.y);
}

// read circle pointer meter
void meterReader::read_meter(std::vector<CropInfo> &crops_meter, std::vector<MeterInfo> &meters)
{
    uint8_t* rect_scale;
    uint8_t* rect_pointer;
    uint8_t* d_rect_scale; // device pointer
    uint8_t* d_rect_pointer; // device pointer
    uint8_t* d_circle_scale; // device pointer
    uint8_t* d_circle_pointer; // device pointer
    int* line_pointer;
    int* line_scale;
    std::vector<int> meter_ids(crops_meter.size(), 0);

    for (int im = 0; im < crops_meter.size(); im++)
    {
        cv::Mat mask_pointer = crops_meter[im].mask_pointer;
        cv::Mat mask_scale = crops_meter[im].mask_scale;

        if (mask_pointer.empty() || mask_scale.empty())
        {
            LOG(WARNING) << "mask empty";
            continue;
        }

        // cv::imwrite("./mask_scale_" + std::to_string(im) + ".png", mask_scale * 255);
        // cv::imwrite("./mask_pointer_" + std::to_string(im) + ".png", mask_pointer * 255);
        
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        std::vector<cv::Point2f> points;
        cv::findContours(mask_scale, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);      

        // cv::Mat cnt_img = cv::Mat::zeros(mask_scale.size(), CV_8UC1);
        // cv::drawContours(cnt_img, contours, -1, cv::Scalar(255), 1);
        // cv::imwrite("./contours_" + std::to_string(im) + ".png", cnt_img);

        for (int ic = 0; ic < contours.size(); ic++)
        {
            for (int ip = 0; ip < contours[ic].size(); ip++)
            {
                points.push_back(contours[ic][ip]);
            }
        }
        
        LOG(INFO) << "points size: " << points.size();

        if (points.size() < 3)
        {
            LOG(WARNING) << "points size < 3";
            continue;
        }
        
        int radius;
        cv::Point center;
        minimum_coverage_circle(points, radius, center);
        LOG(INFO) << "radius: " << radius << " center: " << center;

        if (radius < 40)
        {
            LOG(WARNING) << "radius < 40";
            continue;
        }
        
        // cv::Mat min_cov_cir = mask_scale * 255 + mask_pointer * 255;
        // cv::circle(min_cov_cir, center, radius, cv::Scalar(255), 1);
        // cv::circle(min_cov_cir, center, radius - RECT_HEIGHT, cv::Scalar(255), 1);
        // cv::circle(min_cov_cir, center, 1, cv::Scalar(100), 1);
        // cv::imwrite("./circle_" + std::to_string(im) + ".png", min_cov_cir);

        // auto t1 = clock();
        rect_scale = new uint8_t[RECT_WIDTH * RECT_HEIGHT]; // 360 * 40
        rect_pointer = new uint8_t[RECT_WIDTH * RECT_HEIGHT]; // 360 * 40
        line_scale = new int[RECT_WIDTH]; // 512
        line_pointer = new int[RECT_WIDTH]; // 512
        CUDA_CHECK(cudaMalloc((void**)&d_rect_scale, RECT_WIDTH * RECT_HEIGHT * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_rect_pointer, RECT_WIDTH * RECT_HEIGHT * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_circle_scale, CIRCLE_WIDTH * CIRCLE_HEIGHT * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_circle_pointer, CIRCLE_WIDTH * CIRCLE_HEIGHT * sizeof(uint8_t)));

        CUDA_CHECK(cudaMemcpy(d_circle_scale, mask_scale.data, mask_scale.rows * mask_scale.cols * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_circle_pointer, mask_pointer.data, mask_pointer.rows * mask_pointer.cols * sizeof(uint8_t), cudaMemcpyHostToDevice));

        dim3 block1 = dim3(16, 16);
        dim3 grid1 = dim3((RECT_HEIGHT + block1.x - 1) / block1.x, (RECT_WIDTH + block1.y - 1) / block1.y);

        LOG(INFO) << "kernel circle_to_rect launched with "
                << grid1.x << "x" << grid1.y << "x" << grid1.z << " blocks of "
                << block1.x << "x" << block1.y << "x" << block1.z << " threads";
        
        circle_to_rect<<<grid1, block1>>>(d_circle_pointer, d_rect_pointer, radius, center, RECT_WIDTH, RECT_HEIGHT, CIRCLE_WIDTH, CIRCLE_HEIGHT);
        circle_to_rect<<<grid1, block1>>>(d_circle_scale, d_rect_scale, radius, center, RECT_WIDTH, RECT_HEIGHT, CIRCLE_WIDTH, CIRCLE_HEIGHT);
        // auto t2 = clock();
        // LOG(WARNING) << "circle_to_rect time: " << (t2 - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms";

        CUDA_CHECK(cudaMemcpy(rect_pointer, d_rect_pointer, RECT_HEIGHT * RECT_WIDTH * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(rect_scale, d_rect_scale, RECT_HEIGHT * RECT_WIDTH * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        
        // CPU version
        // auto t1 = clock();
        // circle_to_rect_cpu(mask_pointer.data, rect_pointer, radius, center, RECT_WIDTH, RECT_HEIGHT, CIRCLE_WIDTH, CIRCLE_HEIGHT);
        // circle_to_rect_cpu(mask_scale.data, rect_scale, radius, center, RECT_WIDTH, RECT_HEIGHT, CIRCLE_WIDTH, CIRCLE_HEIGHT);
        // auto t2 = clock();
        // LOG(WARNING) << "circle_to_rect_cpu time: " << (t2 - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms";

        // debug
        // cv::Mat rect_pointer_img = cv::Mat(RECT_HEIGHT, RECT_WIDTH, CV_8UC1, rect_pointer);
        // cv::Mat rect_scale_img = cv::Mat(RECT_HEIGHT, RECT_WIDTH, CV_8UC1, rect_scale);
        // cv::imwrite("./rect_pointer_" + std::to_string(im) + ".png", rect_pointer_img * 255);
        // cv::imwrite("./rect_scale_" + std::to_string(im) + ".png", rect_scale_img * 255);

        // CPU version
        // auto t1 = clock();
        rect_to_line_cpu(rect_pointer, line_pointer, RECT_WIDTH, RECT_HEIGHT);
        rect_to_line_cpu(rect_scale, line_scale, RECT_WIDTH, RECT_HEIGHT);
        // auto t2 = clock();
        // LOG(WARNING) << "rect_to_line_cpu time: " << (t2 - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms";

        // debug
        // for (int i = 0; i < RECT_WIDTH; i++) printf("%d ", line_pointer[i]); printf("\n\n");
        // for (int i = 0; i < RECT_WIDTH; i++) printf("%d ", line_scale[i]); printf("\n");

        std::vector<float> pointer_location;
        std::vector<float> scale_location;

        line_to_location(line_pointer, pointer_location, RECT_WIDTH);
        line_to_location(line_scale, scale_location, RECT_WIDTH);

        // // debug
        // std::cout << "pointer: " << pointer_location.size() << std::endl;
        // for (int i = 0; i < pointer_location.size(); i++) printf("%f ", pointer_location[i]); printf("\n\n");

        // std::cout << "scale: " << scale_location.size() << std::endl;
        // for (int i = 0; i < scale_location.size(); i++) printf("%f ", scale_location[i]); printf("\n\n");

        // consruct MeterInfo and save to meters
        float reading_percent = location_to_reading(pointer_location, scale_location);
        float reading_number = round(reading_percent * METER_RANGES[0] * 100) / 100;
        // std::string meter_reading = std::to_string(reading_number) + " " + METER_UNITS[0];
        char meter_reading[100];
        if (reading_number >= 0.0)
        {
            sprintf(meter_reading, "%.2f %s", reading_number, METER_UNITS[0]);
        }
        else
        {
            sprintf(meter_reading, "N/A");
        }

        MeterInfo meter_info;
        meter_info.rect = crops_meter[im].rect;
        meter_info.camera_id = crops_meter[im].frame_batch_id;
        meter_info.class_id = 0; // meter
        meter_info.class_name = "meter";
        meter_info.meter_id = meter_ids[im]++;
        meter_info.meter_reading = meter_reading;
        meter_info.meter_reading_value = reading_number;
        meters.push_back(meter_info);

        LOG(INFO) << "meter_" + std::to_string(im) + ": " << meter_reading;

        // free memory
        CUDA_CHECK(cudaFree(d_circle_pointer));
        CUDA_CHECK(cudaFree(d_rect_pointer));
        CUDA_CHECK(cudaFree(d_circle_scale));
        CUDA_CHECK(cudaFree(d_rect_scale));
        delete[] rect_pointer;
        delete[] rect_scale;
        delete[] line_pointer;
        delete[] line_scale;

        // LOG_ASSERT(0) << " stop here";
        
    }
}

void meterReader::read_water(std::vector<CropInfo> &crops_water, std::vector<MeterInfo> &meters)
{
    std::vector<int> meter_ids(crops_water.size(), 0);
    for (int im = 0; im < crops_water.size(); im++)
    {
        cv::Rect level_bbox;
        float level_location;
        int level_percent;

        if (crops_water[im].det_objs.size() == 0)
        {
            level_percent = 100;
        }
        else
        {
            level_bbox = crops_water[im].det_objs[0].rect;
            level_location = (level_bbox.y + level_bbox.height / 2.0) / crops_water[im].rect.height;
            level_percent = 100 - round(level_location * 100);

            LOG(INFO) << "level_bbox: " << level_bbox.x << " " << level_bbox.y << " " << level_bbox.width << " " << level_bbox.height;
            LOG(INFO) << "level_location: " << level_location;
        }
        

        MeterInfo meter_info;
        meter_info.rect = crops_water[im].rect;
        meter_info.camera_id = crops_water[im].frame_batch_id;
        meter_info.class_id = 1; // water
        meter_info.class_name = "water";
        meter_info.meter_id = meter_ids[im]++;
        meter_info.meter_reading = std::to_string(level_percent) + " %";
        meter_info.meter_reading_value = level_percent / 100.f;
        meters.push_back(meter_info);

        LOG(INFO) << "water_" + std::to_string(im) + ": " << level_percent << "%";

    }
}

void meterReader::read_number(std::vector<MeterInfo> &meters, std::vector<CropInfo> &crops_meter, std::vector<CropInfo> &crops_water)
{
    std::sort(crops_meter.begin(), crops_meter.end(), 
        [](CropInfo a, CropInfo b) { return a.rect.x == b.rect.x? a.rect.y < b.rect.y: a.rect.x < b.rect.x;});
    std::sort(crops_water.begin(), crops_water.end(), 
        [](CropInfo a, CropInfo b) { return a.rect.x == b.rect.x? a.rect.y < b.rect.y: a.rect.x < b.rect.x;});

    read_meter(crops_meter, meters);
   
    read_water(crops_water, meters);
}