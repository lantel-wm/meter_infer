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
        << "\nCUDA Error:\n"                            \
        << "    File:       " << __FILE__ << "\n"     \
        << "    Line:       " << __LINE__ << "\n"     \
        << "    Error code: " << error_code << "\n"   \
        << "    Error text: "                         \
        << cudaGetErrorString(error_code);            \
} while (0)

#define GET(output, i, j, k) \
    output[(i) * (this->output_bindings[0].dims.d[1] * this->output_bindings[0].dims.d[2]) + (j) * this->output_bindings[0].dims.d[2] + (k)]

#define ARGMAX3(a, b, c) ((a) > (b) ? ((a) > (c) ? 0 : 2) : ((b) > (c) ? 1 : 2))
#define ARGMAX2(a, b) ((a) > (b) ? 0 : 1)

#define DUMP_OBJ_INFO(det_objs) \
    for (int i = 0; i < det_objs.size(); i++) \
    { \
        LOG(INFO) << "name: " << det_objs[i].name << ", class_id: " << det_objs[i].class_id << ", conf: " << det_objs[i].conf << ", rect: " << det_objs[i].rect; \
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

#endif