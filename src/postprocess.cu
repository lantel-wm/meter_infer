#include <opencv2/opencv.hpp>
#include <vector>
#include <cuda_runtime.h>

#include "cublas_v2.h"
#include "glog/logging.h"
#include "yolo.hpp"
#include "stream.hpp"
#include "config.hpp"
#include "common.hpp"

// view mask image of segmentation on device
void view_masks(float* h_ptr, int nobjs, std::vector<DetObject> &det_objs)
{
    int w = 160;
    int h = 160;
    
    for (int iobj = 0; iobj < nobjs; iobj++)
    {
        cv::Mat img(h, w, CV_8UC3, cv::Scalar(0, 0, 0));

        for (int i = 0; i < w * h; i++)
        {
            float conf = h_ptr[iobj + i * nobjs];
            if (conf < 0.80)
                continue;
            // LOG(INFO) << "conf: " << conf << ", iobj: " << iobj << ", i: " << i;
            for (int j = 0; j < 3; j++)
            {
                img.at<cv::Vec3b>(i / w, i % w)[j] = conf * 255;
            }
        }
        cv::Rect bbox = det_objs[iobj].rect;
        bbox.x = round(bbox.x * 160.0 / 640.0);
        bbox.y = round(bbox.y * 160.0 / 640.0);
        bbox.width = round(bbox.width * 160.0 / 640.0);
        bbox.height = round(bbox.height * 160.0 / 640.0);
        cv::rectangle(img, bbox, cv::Scalar(0, 0, 255), 1);

        char mask_savepath[20];
        sprintf(mask_savepath, "./mask/mask_%d.png", iobj);
        cv::imwrite(mask_savepath, img);
    }
    
}

__global__ void sigmoid(float* d_ptr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        d_ptr[idx] = 1.0f / (1.0f + expf(-d_ptr[idx]));
    }
}

void crop_mask(float* masks, int nobjs, std::vector<DetObject> &det_objs, cv::Mat &mask_scale, cv::Mat &mask_pointer)
{
    cv::Mat mask_s(160, 160, CV_8UC1, cv::Scalar(0));
    cv::Mat mask_p(160, 160, CV_8UC1, cv::Scalar(0));
    for (int iobj = 0; iobj < nobjs; iobj++)
    {
        cv::Rect bbox = det_objs[iobj].rect;
        bbox.x = round(bbox.x * 160.0 / 640.0);
        bbox.y = round(bbox.y * 160.0 / 640.0);
        bbox.width = round(bbox.width * 160.0 / 640.0);
        bbox.height = round(bbox.height * 160.0 / 640.0);
        
        // masks: 32 * 160 * 160, col major
        // masks[iobj][i][j] = masks[iobj + i * nobjs + j * nobjs * 160]
        for (int i = 0; i < bbox.width; i++)
        {
            for (int j = 0; j < bbox.height; j++)
            {
                int x = bbox.x + i;
                int y = bbox.y + j;

                int idx = iobj + (x + y * 160) * nobjs;

                if (masks[idx] < 0.8)
                    continue;

                if (det_objs[iobj].class_id == 1)
                {
                    mask_s.at<uint8_t>(y, x) = 1;
                }
                else
                {
                    mask_p.at<uint8_t>(y, x) = 1;
                }
                
            }
        }
    }
    mask_scale = mask_s;
    mask_pointer = mask_p;
}

void Segment::processMask(std::vector<CropInfo> &crops)
{
    // mask_in: n_obj * 32
    // proto: 32 * 160 * 160
    // mask_out = mask_in * proto
    // mask_out: n * 160 * 160

    // use cublas to do matrix multiplication
    
    float *output1 = static_cast<float *>(this->device_ptrs[1]); // proto 8x32x160x160
    LOG(INFO) << "batch_size: " << crops.size();
    int batch_size = crops.size();

    // view_proto(output1);
    // LOG_ASSERT(0) << " stop here";

    for (int ibatch = 0; ibatch < batch_size; ibatch++)
    {
        std::vector<DetObject> det_objs = crops[ibatch].det_objs;

        int nobjs = det_objs.size();
        float *d_mask_in;
        float *d_mask_out;
        size_t mask_in_size = 32 * sizeof(float);
        size_t mask_out_size = nobjs * 160 * 160 * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_mask_in, mask_in_size * nobjs));
        CUDA_CHECK(cudaMalloc(&d_mask_out, mask_out_size));
        for (int iobj = 0; iobj < nobjs; iobj++)
        {
            CUDA_CHECK(cudaMemcpy(d_mask_in + iobj * 32, det_objs[iobj].mask_in, mask_in_size, cudaMemcpyHostToDevice));
        }

        
        // do matrix multiplication: [nobjs * 32] * [32 * 160 * 160] = [nobjs * 160 * 160]
        float alpha = 1.0f;
        float beta = 0.0f;
        LOG(INFO) << "cuBLAS sgemm";
        cublas_status = cublasSgemm_v2(
            cublas_handle,
            CUBLAS_OP_T,
            CUBLAS_OP_T,
            nobjs,
            160 * 160,
            32,
            &alpha,
            d_mask_in,
            32,
            output1 + ibatch * 32 * 160 * 160,
            160 * 160,
            &beta,
            d_mask_out,
            nobjs);
        LOG_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS) << "\nCUBLAS sgemm failed!\n";

        dim3 block1(1024);
        dim3 grid1((nobjs * 160 * 160 + block1.x - 1) / block1.x);

        LOG(INFO) << "sigmoid kernel launched with " << grid1.x << " blocks of "
            << block1.x << " threads";
        sigmoid<<<grid1, block1>>>(d_mask_out, nobjs * 160 * 160);

        float *mask_out = new float[mask_out_size];
        CUDA_CHECK(cudaMemcpy(mask_out, d_mask_out, mask_out_size, cudaMemcpyDeviceToHost));

        LOG(INFO) << "mask_out: " << mask_out[0] << " " << mask_out[1] << " " << mask_out[2];

        // view_masks(mask_out, nobjs, det_objs);

        crop_mask(mask_out, nobjs, det_objs, crops[ibatch].mask_scale, crops[ibatch].mask_pointer);

        // cv::imwrite("./mask_scale.png", crops[ibatch].mask_scale);
        // cv::imwrite("./mask_pointer.png", crops[ibatch].mask_pointer);

        CUDA_CHECK(cudaFree(d_mask_in));
        CUDA_CHECK(cudaFree(d_mask_out));
    }
}