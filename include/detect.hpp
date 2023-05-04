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
    float conf;
    cv::Rect rect; // rect(x, y, w, h), (x, y) is the upperleft point
};

// Store binding information.
struct Binding
{
    size_t size = 1;
    size_t dsize = 1;
    nvinfer1::Dims dims;
    std::string name;
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
        IExecutionContext *context;
        IRuntime *runtime;
        ICudaEngine *engine;
        cudaStream_t stream = nullptr;
        int num_bindings;
        int num_inputs = 0;
        int num_outputs = 0;
        std::vector<Binding> input_bindings;
        std::vector<Binding> output_bindings;
        std::vector<void*> host_ptrs;
        std::vector<void*> device_ptrs;

        cv::Mat processInput(cv::Mat &image); // preprocess the image
        void processOutput(float *output, std::vector<detectResult> &results); // postprocess the image
        // void letterbox(cv::Mat &image);
        void nonMaxSupression(std::vector<detectResult> &results); // non-maximum suppression
        float iou(cv::Rect &rect1, cv::Rect &rect2); // calculate the IOU of two rectangles

    public:
        Detect(std::string const &engine_path); // load the engine
        ~Detect(); // unload the engine
        void Infer(cv::Mat &image, std::vector<detectResult> &results); // detect the image
};

#endif