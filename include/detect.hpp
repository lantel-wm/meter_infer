#ifndef _DETECT_HPP_
#define _DETECT_HPP_

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "NvInfer.h"

using namespace nvinfer1;

// Store detection results.
struct detectResult
{
    std::string name;
    int class_id;
    float confidence;
    cv::Rect rect; // rect(x, y, w, h), (x, y) is the upperleft point
};

// Run image detection.
// Example:
//      Detect detect("yolov8n_batch8.trt");
//      cv::Mat image = cv::imread("data/images/60.png");
//      detect.Infer(image);
class Detect
{
    private:
        int width; // input width
        int height; // input height
        cv::Mat M; // affine transformation matrix
        cv::Mat IM; // inverse affine transformation matrix
        std::string engine_path;
        IRuntime *runtime;
        ICudaEngine *engine;
        IExecutionContext *context;

        cv::Mat processInput(cv::Mat &image); // preprocess the image
        void processOutput(float *output, std::vector<detectResult> &results); // postprocess the image
        // void letterbox(cv::Mat &image);
        // void nms(std::vector<detectResult> &results, float nms_threshold); // non-maximum suppression

    public:
        Detect(std::string const &engine_path); // load the engine
        ~Detect(); // unload the engine
        void Infer(cv::Mat &image, std::vector<detectResult> &results); // detect the image
};

#endif