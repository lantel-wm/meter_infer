#include <opencv2/opencv.hpp>
#include <vector>
#include <cuda_runtime.h>

#include "cublas_v2.h"
#include "glog/logging.h"
#include "yolo.hpp"
#include "stream_to_img.hpp"
#include "config.hpp"
#include "common.hpp"


void Segment::processMask(std::vector<CropInfo> &crops)
{
    // mask_in: n * 32
    // proto: 32 * 160 * 160
    // mask_out = mask_in * proto
    // mask_out: n * 160 * 160

    float* output1 = static_cast<float*>(this->device_ptrs[0]); // proto
    LOG(INFO) << "output1: " << sizeof(output1);
    LOG_ASSERT(0) << "stop";
    int ibatch = 0;

    for (auto &crop : crops)
    {
        std::vector<DetObject> det_objs = crop.det_objs;

        int n_objs = det_objs.size();
        float* d_mask_in;
        float* d_mask_out;
        size_t mask_in_size = 32 * sizeof(float);
        size_t mask_out_size = n_objs * 160 * 160 * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_mask_in, mask_in_size * n_objs));
        CUDA_CHECK(cudaMalloc(&d_mask_out, mask_out_size));
        for (auto &obj : det_objs)
        {
            CUDA_CHECK(cudaMemcpy(d_mask_in + ibatch * 32, obj.mask_in, mask_in_size, cudaMemcpyHostToDevice));

        }

        CUDA_CHECK(cudaFree(d_mask_in));
        CUDA_CHECK(cudaFree(d_mask_out));

        ibatch++;
    }
}