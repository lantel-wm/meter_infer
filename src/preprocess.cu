#include <opencv2/opencv.hpp>
#include <vector>

#include "glog/logging.h"
#include "detect.hpp"
#include "config.hpp"
#include "common.hpp"

using namespace google;

__device__ void affine_project(float *mat, int x, int y, float *proj_x, float *proj_y)
{
    // matrix
    // m0, m1, m2
    // m3, m4, m5
    *proj_x = mat[0] * x + mat[1] * y + mat[2];
    *proj_y = mat[3] * x + mat[4] * y + mat[5];
}

// warp affine transformation by bilinear interpolation
__global__ void warp_affine(
    uint8_t *src, int src_line_size, int src_width, int src_height,
    uint8_t *dst, int dst_line_size, int dst_width, int dst_height,
    uint8_t fill_value, AffineMatrix M)
{
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;

    if (dx >= dst_width || dy >= dst_height)
        return;

    float c0 = fill_value, c1 = fill_value, c2 = fill_value;
    float src_x = 0;
    float src_y = 0;
    affine_project(M.inv_mat, dx, dy, &src_x, &src_y);

    if (src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height)
    {
        // out of range
        // when src_x < -1，high_x < 0，out of range
        // when src_x >= -1，high_x >= 0，in range
    }
    else
    {
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_values[] = {fill_value, fill_value, fill_value};
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t *v1 = const_values;
        uint8_t *v2 = const_values;
        uint8_t *v3 = const_values;
        uint8_t *v4 = const_values;
        if (y_low >= 0)
        {
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }

        if (y_high < src_height)
        {
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }

        c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
        c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
        c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
    }

    uint8_t *pdst = dst + dy * dst_line_size + dx * 3;
    pdst[0] = c0;
    pdst[1] = c1;
    pdst[2] = c2;
}

void Detect::preprocess(std::vector<cv::Mat> &images)
{
   for (auto &src : images)
   {
        uint8_t* d_ptr_src; // device pointer for src image
        uint8_t* d_pre_dst; // device pointer for dst image
        int src_w = src.cols; // src image width
        int src_h = src.rows; // src image height
        int dst_w = this->input_width; // dst image width
        int dst_h = this->input_height; // dst image height
        size_t src_size = src_w * src_h * 3 * sizeof(uint8_t); // src image size
        size_t dst_size = dst_w * dst_h * 3 * sizeof(uint8_t); // dst image size

        CUDA_CHECK(cudaMalloc((uint8_t **)&d_ptr_src, src_size)); 
        CUDA_CHECK(cudaMalloc((uint8_t **)&d_pre_dst, dst_size));
        CUDA_CHECK(cudaMemcpy(d_ptr_src, src.data, src_size, cudaMemcpyHostToDevice));
        
        // compute affine tranformation matrix
        (this->affine_matrix).compute(cv::Size(src_w, src_h), cv::Size(dst_w, dst_h));

        dim3 block(32, 32);
        dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

        LOG(INFO) << "warp_affine kernel launch with " 
            << grid.x << "x" << grid.y << " blocks of " 
            << block.x << "x" << block.y << " threads";

        // do letterbox transformation on src image
        // src: [src_h, src_w, 3], dst: [dst_h, dst_w, 3]
        warp_affine<<<grid, block, 0, nullptr>>>(
            d_ptr_src, src_w * 3, src_w, src_h,
            d_pre_dst, dst_w * 3, dst_w, dst_h,
            114, this->affine_matrix
        );
        
   }
}