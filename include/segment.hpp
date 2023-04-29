#ifndef _SEGMENT_HPP_
#define _SEGMENT_HPP_

#include <opencv2/opencv.hpp>
#include <string>

#include "NvInfer.h"
#include "glog/logging.h"

using namespace nvinfer1;

struct Masks
{
    cv::Mat pointer_mask;
    cv::Mat scale_mask;
};

class Segment
{
    protected:
        std::string engine_path;
        IRuntime *runtime;
        ICudaEngine *engine;
        IExecutionContext *context;

    public:
        Segment(std::string const &engine_path); // load the engine
        ~Segment(); // unload the engine
        void preprocess(cv::Mat &image); // preprocess the image
        void postprocess(cv::Mat &image); // postprocess the image
        virtual void infer(cv::Mat &image); // segment the image
};

#endif