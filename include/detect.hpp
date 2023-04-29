#ifndef _DETECT_HPP_
#define _DETECT_HPP_

#include <opencv2/opencv.hpp>
#include <string>

#include "NvInfer.h"
#include "glog/logging.h"

using namespace nvinfer1;

struct detectResult
{
    std::string name;
    int class_id;
    float confidence;
    cv::Rect rect; // rect(x, y, w, h), (x, y) is the upperleft point
};

class Detect
{
    protected:
        std::string engine_path;
        IRuntime *runtime;
        ICudaEngine *engine;
        IExecutionContext *context;

    public:
        Detect(std::string const &engine_path); // load the engine
        ~Detect(); // unload the engine
        void preprocess(cv::Mat &image); // preprocess the image
        void postprocess(cv::Mat &image); // postprocess the image
        virtual void infer(cv::Mat &image); // detect the image
};

#endif