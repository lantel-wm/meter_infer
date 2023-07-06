#ifndef _COMMON_HPP_
#define _COMMON_HPP_

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <vector>
#include <string>

#include "glog/logging.h"

using namespace nvinfer1;

#define CUDA_CHECK(call)                              \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    LOG_ASSERT(error_code == cudaSuccess)             \
        << "\nCUDA Error:\n"                          \
        << "    File:       " << __FILE__ << "\n"     \
        << "    Line:       " << __LINE__ << "\n"     \
        << "    Error code: " << error_code << "\n"   \
        << "    Error text: "                         \
        << cudaGetErrorString(error_code);            \
} while (0)

#define GET(output, i, j, k, d0, d1, d2) \
    output[(i) * ((d1) * (d2)) + (j) * (d2) + (k)]


// get value from 3D array
// a is a pointer to the array, i, j, k are the index of the array, h, w, c are the height, width and channel of the array
// the array is stored in the format of HWC
#define GET3(a, i, j, k, h, w, c) \
    a[(i) * (h * w) + (j) * (w) + (k)]

// get value from 4D array
// a is a pointer to the array, i, j, k, l are the index of the array, n, c, h, w are the batch size, channel, height and width of the array
// the array is stored in the format of NCHW
#define GET4(a, i, j, k, l, n, c, h, w) \
    a[(i) * (c * h * w) + (j) * (h * w) + (k) * (w) + (l)]

#define ARGMAX3(a, b, c) ((a) > (b) ? ((a) > (c) ? 0 : 2) : ((b) > (c) ? 1 : 2))
#define ARGMAX2(a, b) ((a) > (b) ? 0 : 1)

#define DUMP_OBJ_INFO(a) \
    for (int i = 0; i < (a).size(); i++) \
    { \
        LOG(INFO) << "name: " << (a)[i].name << ", class_id: " << (a)[i].class_id << ", conf: " << (a)[i].conf << ", rect: " << (a)[i].rect; \
    }

#define DUMP_VECTOR(a) \
    for (int i = 0; i < a.size(); i++) \
    { \
        LOG(INFO) << a[i]; \
    }


// set val to min if val < min, set val to max if val > max
inline static float clamp(float val, float min, float max)
{
	return val > min ? (val < max ? val : max) : min;
}

inline int get_size_by_dims(const nvinfer1::Dims& dims)
{
	int size = 1;
	for (int i = 0; i < dims.nbDims; i++)
	{
		size *= dims.d[i];
	}
	return size;
}

inline int type_to_size(const nvinfer1::DataType& dataType)
{
	switch (dataType)
	{
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kBOOL:
            return 1;
        default:
            return 4;
	}
}

// view images on device to help debugging
inline void view_device_img(uint8_t* d_ptr, size_t size, int w, int h, std::string name)
{
    LOG(INFO) << "viewing device image " << name;
    cv::Mat img(w, h, CV_8UC3);
    CUDA_CHECK(cudaMemcpy(img.data, d_ptr, size, cudaMemcpyDeviceToHost));
    cv::imwrite(name + ".jpg", img);
    LOG(INFO) << "device image" << name << "saved";
}

// view letterbox images after warpaffine on device to help debugging
inline void view_device_input_img(float* d_ptr, size_t size, int w, int h, std::string name)
{
    LOG(INFO) << "viewing device input image " << name;
    cv::Mat img(w, h, CV_8UC3);
    CUDA_CHECK(cudaMemcpy(img.data, d_ptr, size, cudaMemcpyDeviceToHost));
    cv::imwrite(name + ".jpg", img);
    LOG(INFO) << "device input image" << name << "saved";
}

// view image batch on device to help debugging
inline void view_device_input_img_batch(uint8_t* d_ptr, int n, int c, int h, int w, std::string name)
{
    LOG(INFO) << "viewing device image batch " << name;
    cv::Mat img(h * 2, w * 4, CV_8UC3);
    LOG(INFO) << "img size: " << img.size();
    size_t size = n * c * w * h * sizeof(uint8_t);
    uint8_t* h_ptr = new uint8_t[size];
    CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost));
    const int loc[8][2] = {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 0}, {1, 1}, {1, 2}, {1, 3}};
    // n * h * w * c
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < h; j++)
        {
            for (int k = 0; k < w; k++)
            {
                int ox = loc[i][0] * h + j;
                int oy = loc[i][1] * w + k;
                for (int l = 0; l < c; l++)
                {
                    img.at<cv::Vec3b>(ox, oy)[l] = h_ptr[i * h * w * c + j * w * c + k * c + l];   
                }
            }
        }
    }
    cv::imwrite(name + ".jpg", img);
    LOG(INFO) << "device image" << name << "saved";
}

// view batch images on device to help debugging
inline void view_device_batch_img(float* d_ptr, int n, int c, int w, int h, std::string name)
{
    LOG(INFO) << "viewing device batch image " << name;
    cv::Mat img(h * 2, w * 4, CV_8UC3);
    LOG(INFO) << "img size: " << img.size();
    size_t size = n * c * w * h * sizeof(float);
    float* h_ptr = new float[size];
    CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost));
    const int loc[8][2] = {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 0}, {1, 1}, {1, 2}, {1, 3}};
    for (int i = 0; i < n; i++)
    {
        // LOG(INFO) << "batch: " << i << ", location: " << loc[i][0] * w << ", " << loc[i][1] * h;
        for (int j = 0; j < c; j++)
        {
            for (int k = 0; k < h; k++)
            {
                for (int l = 0; l < w; l++)
                {
                    int ox = loc[i][0] * h + k;
                    int oy = loc[i][1] * w + l;
                    float dn = clamp(GET4(h_ptr, i, j, k, l, n, c, h, w) * 255, 0, 255);
                    img.at<cv::Vec3b>(ox, oy)[2 - j] = dn; // RGB -> BGR
                }
            }
        }
    }
    cv::imwrite(name + ".jpg", img);
    LOG(INFO) << "device batch image" << name << "saved";
}

// view proto image of segmentation on device to help debugging
inline void view_proto(float* d_ptr)
{
    int w = 160;
    int h = 160;
    cv::Mat img(h, w, CV_8UC3);

    for (int ipro = 0; ipro < 32; ipro++)
    {
        float* h_ptr = new float[w * h];
        CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr + ipro * w * h, w * h * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < w * h; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                img.at<cv::Vec3b>(i / w, i % w)[j] = clamp(h_ptr[i] * 255, 0, 255);
            }
        }
        char proto_savepath[20];
        sprintf(proto_savepath, "./proto/proto_%d.jpg", ipro);
        cv::imwrite(proto_savepath, img);
    }
}

#endif