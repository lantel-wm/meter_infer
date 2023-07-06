#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <random>
#include <math.h>

#include "glog/logging.h"
#include "meter_reader.hpp"
#include "yolo.hpp"
#include "config.hpp"
#include "common.hpp"

#define PI 3.1415926f

__global__ void circle_to_rect(uint8_t* circle, uint8_t* rect, int radius, cv::Point &center,
    int rect_width, int rect_height, int circle_width, int circle_height)
{
    int d_rho = blockIdx.x * blockDim.x + threadIdx.x;
    int theta = blockIdx.y * blockDim.y + threadIdx.y;

    if (d_rho >= rect_height || theta >= rect_width)
        return;

    int rho = d_rho + radius - rect_height;
    int x = round(center.x + rho * cos(theta * PI / 180.0f));
    int y = round(center.y + rho * sin(theta * PI / 180.0f));

    if (x < 0 || x >= circle_width || y < 0 || y >= circle_height)
    {
        rect[d_rho * rect_width + theta] = 0;
        return;
    }

    rect[d_rho * rect_width + theta] = circle[y * circle_width + x];
    
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
        
        // cv::Mat min_cov_cir = mask_scale + mask_pointer;
        // cv::circle(min_cov_cir, center, radius, cv::Scalar(255), 1);
        // cv::circle(min_cov_cir, center, radius - 40, cv::Scalar(255), 1);
        // cv::circle(min_cov_cir, center, 1, cv::Scalar(100), 1);
        // cv::imwrite("./circle.png", min_cov_cir);

        CUDA_CHECK(cudaMemcpy(d_circle_scale, mask_scale.data, mask_scale.rows * mask_scale.cols * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_circle_pointer, mask_pointer.data, mask_pointer.rows * mask_pointer.cols * sizeof(uint8_t), cudaMemcpyHostToDevice));

        dim3 block = dim3(32, 32);
        dim3 grid = dim3((RECT_HEIGHT + block.x - 1) / block.x, (RECT_WIDTH + block.y - 1) / block.y);

        LOG(INFO) << "kernel circle_to_rect launched with "
                << grid.x << "x" << grid.y << "x" << grid.z << " blocks of "
                << block.x << "x" << block.y << "x" << block.z << " threads";
        circle_to_rect<<<grid, block>>>(d_circle_pointer, d_rect_pointer, radius, center, RECT_WIDTH, RECT_HEIGHT, CIRCLE_WIDTH, CIRCLE_HEIGHT);
        circle_to_rect<<<grid, block>>>(d_circle_scale, d_rect_scale, radius, center, RECT_WIDTH, RECT_HEIGHT, CIRCLE_WIDTH, CIRCLE_HEIGHT);

        CUDA_CHECK(cudaMemcpy(rect_pointer, d_rect_pointer, RECT_HEIGHT * RECT_WIDTH * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(rect_scale, d_rect_scale, RECT_HEIGHT * RECT_WIDTH * sizeof(uint8_t), cudaMemcpyDeviceToHost));

        cv::Mat rect_pointer_img = cv::Mat(RECT_HEIGHT, RECT_WIDTH, CV_8UC1, rect_pointer);
        cv::Mat rect_scale_img = cv::Mat(RECT_HEIGHT, RECT_WIDTH, CV_8UC1, rect_scale);
        cv::imwrite("./rect_pointer.png", rect_pointer_img);
        cv::imwrite("./rect_scale.png", rect_scale_img);
        LOG_ASSERT(0) << "stop here";
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