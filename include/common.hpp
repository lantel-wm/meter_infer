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