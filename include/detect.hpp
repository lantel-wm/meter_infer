#ifndef _DETECT_HPP_
#define _DETECT_HPP_

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "NvInfer.h"

using namespace nvinfer1;

// Store detection results.
struct DetObject
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
        int input_width; // input width
        int input_height; // input height
        int image_width; // output width
        int image_height; // output height
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

        void letterbox(const cv::Mat& image, cv::Mat& out); // make letterbox for the image
        void postprocess(std::vector<DetObject> &det_objs); // postprocess the image
        void makePipe(bool warmup);
        void copyFromMat(cv::Mat &nchw);
        void infer();

        void nonMaxSuppression(std::vector<DetObject> &det_objs); // non-maximum suppression
        float iou(const cv::Rect rect1, const cv::Rect rect2); // calculate the IOU of two rectangles

    public:
        Detect(std::string const &engine_path); // load the engine
        ~Detect(); // unload the engine
        void detect(cv::Mat &image, std::vector<DetObject> &results); // detect the image
        void engineInfo(); // print the engine information
};

#endif