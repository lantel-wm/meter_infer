#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <random>
#include <math.h>
#include <iostream>

#include "glog/logging.h"
#include "meter_reader.hpp"
#include "yolo.hpp"
#include "config.hpp"
#include "common.hpp"

#define PI 3.1415926f

float location_to_reading(std::vector<float> p_loc, std::vector<float> s_loc)
// return a float range in [0, 1]
{
    int num_scales = s_loc.size();
    
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

void line_to_location(int *line, std::vector<float> &location, int width)
{
    float index_buffer[width];
    int ib_cur = 0; // pointer to index_buffer
    bool ascending = true;
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

void meterReader::minimum_coverage_circle(std::vector<cv::Point> &points, int &radius, cv::Point &center)
{
    std::random_shuffle(points.begin(), points.end());
    int n = points.size();
    float radius_f = 0; // use float instead of int to avoid error in calculation
    center = points[0];
    radius_f = 0;

    for (int i = 1; i < n; i++)
    {
        if (cv::norm(points[i] - center) <= radius_f) // points[i] is not in circle
            continue;
        
        center = points[i];
        radius_f = 0;
        for (int j = 0; j < i; j++)
        {
            if (cv::norm(points[j] - center) <= radius_f) // points[j] is not in circle
                continue;

            center = (points[i] + points[j]) / 2;
            radius_f = cv::norm(points[i] - points[j]) / 2;
            for (int k = 0; k < j; k++)
            {
                if (cv::norm(points[k] - center) <= radius_f) // points[k] is not in circle
                    continue;
                
                // calculate the center and radius of the circle passing through points i, j, k
                cv::Point2f a = points[i] - points[j]; // AB
                cv::Point2f b = points[i] - points[k]; // AC
                cv::Point2f p = (points[i] + points[j]) / 2; // midpoint of AB
                cv::Point2f q = (points[i] + points[k]) / 2; // midpoint of AC
                cv::Point2f v = cv::Point2f(-a.y, a.x); // vector perpendicular to AB
                cv::Point2f w = cv::Point2f(-b.y, b.x); // vector perpendicular to AC
                float t1 = w.cross(p - q) / v.cross(w); // t1 = (q - p) x w / v x w
                center = p + t1 * v; // center = p + t1 * v
                radius_f = cv::norm(points[i] - center);
            }
        }
    }

    radius = (int)radius_f;
}

void meterReader::read_meter(std::vector<CropInfo> &crops_meter, std::vector<MeterInfo> &meters)
{
    for (int im = 0; im < crops_meter.size(); im++)
    {
        cv::Mat mask_pointer = crops_meter[im].mask_pointer;
        cv::Mat mask_scale = crops_meter[im].mask_scale;

        // cv::Mat edges;
        // cv::Canny(mask_scale, edges, 50, 150);
        // cv::imwrite("./edges.png", edges);
        
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        std::vector<cv::Point> points;
        cv::findContours(mask_scale, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);      

        // cv::Mat cnt_img = cv::Mat::zeros(mask_scale.size(), CV_8UC1);
        // cv::drawContours(cnt_img, contours, -1, cv::Scalar(255), 1);
        // cv::imwrite("./contours.png", cnt_img);

        for (int ic = 0; ic < contours.size(); ic++)
        {
            for (int ip = 0; ip < contours[ic].size(); ip++)
            {
                points.push_back(contours[ic][ip]);
            }
        }
        
        LOG(INFO) << "points size: " << points.size();

        int radius;
        cv::Point center;
        minimum_coverage_circle(points, radius, center);
        LOG(INFO) << "radius: " << radius << " center: " << center;
        
        cv::Mat min_cov_cir = mask_scale + mask_pointer;
        cv::circle(min_cov_cir, center, radius, cv::Scalar(255), 1);
        cv::circle(min_cov_cir, center, radius - RECT_HEIGHT, cv::Scalar(255), 1);
        cv::circle(min_cov_cir, center, 1, cv::Scalar(100), 1);
        cv::imwrite("./circle.png", min_cov_cir);

        // auto t1 = clock();
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
        // cv::imwrite("./rect_pointer.png", rect_pointer_img);
        // cv::imwrite("./rect_scale.png", rect_scale_img);
        // LOG_ASSERT(0) << "stop here";

        // CPU version
        // auto t1 = clock();
        rect_to_line_cpu(rect_pointer, line_pointer, RECT_WIDTH, RECT_HEIGHT);
        rect_to_line_cpu(rect_scale, line_scale, RECT_WIDTH, RECT_HEIGHT);
        // auto t2 = clock();
        // LOG(WARNING) << "rect_to_line_cpu time: " << (t2 - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms";

        // debug
        for (int i = 0; i < RECT_WIDTH; i++) printf("%d ", line_pointer[i]); printf("\n\n");
        for (int i = 0; i < RECT_WIDTH; i++) printf("%d ", line_scale[i]); printf("\n");

        std::vector<float> pointer_location;
        std::vector<float> scale_location;

        line_to_location(line_pointer, pointer_location, RECT_WIDTH);
        line_to_location(line_scale, scale_location, RECT_WIDTH);

        std::cout << "pointer: " << pointer_location.size() << std::endl;
        for (int i = 0; i < pointer_location.size(); i++) printf("%f ", pointer_location[i]); printf("\n\n");

        std::cout << "scale: " << scale_location.size() << std::endl;
        for (int i = 0; i < scale_location.size(); i++) printf("%f ", scale_location[i]); printf("\n\n");

        float reading_percent = location_to_reading(pointer_location, scale_location);
        float reading_number = reading_percent * METER_RANGES[0];
        std::string meter_reading = std::to_string(reading_number) + " " + METER_UNITS[0];

        MeterInfo meter_info;
        meter_info.class_id = 0; // meter
        meter_info.class_name = "meter";
        meter_info.meter_id = im;
        meter_info.meter_reading = meter_reading;
        meters.push_back(meter_info);

        LOG(WARNING) << meter_reading;

        LOG_ASSERT(0) << " stop here";
        
    }
}

void meterReader::read_number(std::vector<MeterInfo> &meters)
{
    std::sort(crops_meter.begin(), crops_meter.end(), 
        [](CropInfo a, CropInfo b) { return a.rect.x == b.rect.x? a.rect.y < b.rect.y: a.rect.x < b.rect.x;});
    std::sort(crops_water.begin(), crops_water.end(), 
        [](CropInfo a, CropInfo b) { return a.rect.x == b.rect.x? a.rect.y < b.rect.y: a.rect.x < b.rect.x;});

    read_meter(crops_meter, meters);
    // read_water(crops_water, meters);
}