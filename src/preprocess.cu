// preprocess for detection
#include <opencv2/opencv.hpp>
#include <vector>

#include "glog/logging.h"
#include "yolo.hpp"
#include "stream_to_img.hpp"
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
    int n = blockDim.z * blockIdx.z + threadIdx.z; // ibatch
    int offset_dst = n * dst_width * dst_height * 3;
    int offset_src = n * src_width * src_height * 3;

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
                v1 = src + y_low * src_line_size + x_low * 3 + offset_src;

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3 + offset_src;
        }

        if (y_high < src_height)
        {
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3 + offset_src;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3 + offset_src;
        }

        c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
        c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
        c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
    }

    uint8_t *pdst = dst + dy * dst_line_size + dx * 3 + offset_dst;
    pdst[0] = c0;
    pdst[1] = c1;
    pdst[2] = c2;
}

// [h, w, c] -> [c, h, w]
// 0...255 -> 0...1
// BGR -> RGB
__global__ void blobFromImage(uint8_t* input, float* output, int h, int w, int c, int n)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < h && y < w && z < c)
    {
        for (int ibatch = 0; ibatch < n; ibatch++)
        {
            int input_idx = x * (c * w) + y * c + (2 - z);
            int output_idx = ibatch * (w * h * c) + z * (w * h) + x * w + y;
            output[output_idx] = input[input_idx] / 255.f;
        }
    }
}


void Detect::preprocess(std::vector<FrameInfo> &images)
{
    int batch_size = images.size();
    LOG_ASSERT(batch_size) << "images is empty";

    uint8_t *d_ptr_src;                                    // device pointer for src image
    uint8_t *d_ptr_dst;                                    // device pointer for dst image
    int src_w = images[0].frame.cols;                                  // src image width
    int src_h = images[0].frame.rows;                                  // src image height
    int dst_w = this->input_width;                         // dst image width
    int dst_h = this->input_height;                        // dst image height
    size_t src_size = src_w * src_h * 3 * sizeof(uint8_t); // src image size
    size_t dst_size = batch_size * dst_w * dst_h * 3 * sizeof(uint8_t); // dst image size

    LOG(INFO) << "src_w: " << src_w << ", src_h: " << src_h << ", dst_w: " << dst_w << ", dst_h: " << dst_h;

    CUDA_CHECK(cudaMalloc((uint8_t **)&d_ptr_src, src_size * batch_size));
    CUDA_CHECK(cudaMalloc((uint8_t **)&d_ptr_dst, dst_size));

    // copy a batch of src images to device
    int ibatch = 0;
    for (auto &src : images)
    {
        CUDA_CHECK(cudaMemcpy(d_ptr_src + ibatch * src_w * src_h * 3, src.frame.data, src_size, cudaMemcpyHostToDevice));
        ibatch++;
    }

    // compute affine tranformation matrix
    (this->affine_matrix).compute(cv::Size(src_w, src_h), cv::Size(dst_w, dst_h));

    dim3 block1(8, 8, batch_size);
    dim3 grid1((dst_w + block1.x - 1) / block1.x, (dst_h + block1.y - 1) / block1.y);

    LOG(INFO) << "warp_affine kernel launched with "
                << grid1.x << "x" << grid1.y << "x" << grid1.z << " blocks of "
                << block1.x << "x" << block1.y << "x" << block1.z << " threads";

    // do letterbox transformation on src image
    // src: [src_h, src_w, 3], dst: [dst_h, dst_w, 3]
    warp_affine<<<grid1, block1>>>(
        d_ptr_src, src_w * 3, src_w, src_h,
        d_ptr_dst, dst_w * 3, dst_w, dst_h,
        114, this->affine_matrix);
    
    // warp affine test code, currently no bug
    // view_device_input_img_batch(d_ptr_dst, batch_size, 3, dst_h, dst_w, "dst");
    // LOG_ASSERT(0) << "stop here";

    dim3 block2(16, 16, 3);
    dim3 grid2((dst_w + block2.x - 1) / block2.x, (dst_h + block2.y - 1) / block2.y, (3 + block2.z - 1) / block2.z);

    LOG(INFO) << "blobFromImage kernel launched with "
                << grid2.x << "x" << grid2.y << "x" << grid2.z << " blocks of "
                << block2.x << "x" << block2.y << "x" << block2.z << " threads";

    // TODO: fix bug here
    // TODO: flip the channel order
    blobFromImage<<<grid2, block2>>>(
        d_ptr_dst, (float*)this->device_ptrs[0],
        dst_h, dst_w, 3, batch_size
    );

    CUDA_CHECK(cudaFree(d_ptr_src));
    CUDA_CHECK(cudaFree(d_ptr_dst));

    // blobFromImage test code, currently no bug
    // view_device_batch_img((float*)this->device_ptrs[0], batch_size, 3, this->input_width, this->input_height, "input");
    // LOG_ASSERT(0) << "stop here";
}

void Segment::preprocess(std::vector<CropInfo> &crops)
{
    int batch_size = crops.size();
    LOG_ASSERT(batch_size) << "crops is empty";

    uint8_t *d_ptr;
    int w = this->input_width;
    int h = this->input_height;

    size_t size = w * h * 3 * sizeof(uint8_t);

    CUDA_CHECK(cudaMalloc((uint8_t **)&d_ptr, batch_size * size));

    int ibatch = 0;
    for(auto crop_info : crops)
    {
        // cv::resize(crop_info.crop, crop_info.crop, cv::Size(this->input_width, this->input_height));   
        LOG(INFO) << "crop size" << crop_info.crop.size();
        CUDA_CHECK(cudaMemcpy(d_ptr + ibatch * w * h * 3, crop_info.crop.data, size, cudaMemcpyHostToDevice));
        ibatch++;
    }
    
    dim3 block(16, 16, 3);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, (3 + block.z - 1) / block.z);

    LOG(INFO) << "blobFromImage kernel launched with "
                << grid.x << "x" << grid.y << "x" << grid.z << " blocks of "
                << block.x << "x" << block.y << "x" << block.z << " threads";

    blobFromImage<<<grid, block>>>(
        d_ptr, (float*)this->device_ptrs[0], 
        h, w, 3, batch_size
    );

    // blobFromImage test code, currently no bug
    // view_device_batch_img((float*)this->device_ptrs[0], batch_size, 3, this->input_width, this->input_height, "input_seg");
    // LOG_ASSERT(0) << "stop here";

    // CUDA_CHECK(cudaFree(d_ptr_src));
    // CUDA_CHECK(cudaFree(d_ptr_dst));
}