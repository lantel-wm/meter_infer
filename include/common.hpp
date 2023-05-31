#ifndef _COMMON_HPP_
#define _COMMON_HPP_

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <string>

#include "glog/logging.h"

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

#define GET(output, i, j, k) \
    output[(i) * (this->output_bindings[0].dims.d[1] * this->output_bindings[0].dims.d[2]) + (j) * this->output_bindings[0].dims.d[2] + (k)]

// get value from 3D array
// a is a pointer to the array, i, j, k are the index of the array, h, w, c are the height, width and channel of the array
// the array is stored in the format of HWC
#define GET3(a, i, j, k, h, w, c) \
    a[(i) * (h * w) + (j) * (w) + (k)]

// get value from 4D array
// a is a pointer to the array, i, j, k, l are the index of the array, n, c, h, w are the batch size, channel, height and width of the array
// the array is stored in the format of NCHW
#define GET4(a, i, j, k, l, n, c, h, w) \
    a[(i) * (n * c * h) + (j) * (c * h) + (k) * (h) + (l)]

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

// view input images on device to help debugging
inline void view_device_input_img(float* d_ptr, size_t size, int w, int h, std::string name)
{
    LOG(INFO) << "viewing device input image " << name;
    cv::Mat img(w, h, CV_8UC3);
    float* h_ptr
    CUDA_CHECK(cudaMemcpy(img.data, d_ptr, size, cudaMemcpyDeviceToHost));
    cv::imwrite(name + ".jpg", img);
    LOG(INFO) << "device input image" << name << "saved";
}

#endif