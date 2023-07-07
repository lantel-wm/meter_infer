#ifndef _YOLO_HPP_
#define _YOLO_HPP_

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "NvInfer.h"
#include "common.hpp"
#include "stream.hpp"

using namespace nvinfer1;

// Store binding information.
struct Binding
{
    size_t size = 1;
    size_t dsize = 1;
    nvinfer1::Dims dims;
    std::string name;
};

// Store detection results.
struct DetObject
{
    std::string name;
    int batch_id;
    int class_id;
    float conf;
    float reading;
    float* mask_in;
    cv::Rect rect; // rect(x, y, w, h), (x, y) is the upperleft point
    cv::Mat mask; // 160x160 image, mask of the detected object
    std::string class_name;
    std::string meter_reading;
};

struct FrameInfo
{
    cv::Mat frame;
    std::string info;
    std::vector<DetObject> det_objs;
};

struct CropInfo
{
    cv::Mat crop; // 640x640
    cv::Mat mask_pointer; // 160x160
    cv::Mat mask_scale; // 160x160
    cv::Rect rect; // rect(x, y, w, h), (x, y) is the upperleft point
    int class_id;
    int det_batch_id; // detection batch id
    std::vector<DetObject> det_objs; // scales or pointer or water level
};

struct MeterInfo
{
    int meter_id; // meter identifier in the frame, sorted by the coordinate of the upperleft point
    cv::Rect rect; // rect(x, y, w, h), (x, y) is the upperleft point
    int class_id; // 0: meter, 1: water
    int det_batch_id; // detection batch id
    std::string class_name; // meter, water
    std::string meter_reading; // e.g.: 2.3kPa, 66%
};

// logger used in TensorRT
class Logger : public ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            // std::cout << msg << std::endl;
            LOG(WARNING) << msg;
    }
};


// Store affine transformation matrix.
struct AffineMatrix
{
    // m0, m1, m2;
    // m3, m4, m5
    float mat[6];     // src to dst(network), 2x3 matrix ==> M
    float inv_mat[6]; // dst(network) to src, 2x3 matrix ==> IM

    // solve the M and IM matrix
    void compute(const cv::Size &src, const cv::Size &dst)
    {
        float scale_x = dst.width / (float)src.width;
        float scale_y = dst.height / (float)src.height;

        float scale = MIN(scale_x, scale_y);
        /*
        M = [
        scale,    0,     -scale * from.width  * 0.5 + to.width  * 0.5
        0,     scale,    -scale * from.height * 0.5 + to.height * 0.5
        ]
        */
        /*
            - 0.5 is to make the center aligned.
        */
        mat[0] = scale;
        mat[1] = 0;
        mat[2] =
            -scale * src.width * 0.5 + dst.width * 0.5 + scale * 0.5 - 0.5;

        mat[3] = 0;
        mat[4] = scale;
        mat[5] =
            -scale * src.height * 0.5 + dst.height * 0.5 + scale * 0.5 - 0.5;

        inv_mat[0] = 1 / mat[0];
        inv_mat[1] = 0;
        inv_mat[2] = -mat[2] / mat[0];
        inv_mat[3] = 0;
        inv_mat[4] = 1 / mat[4];
        inv_mat[5] = -mat[5] / mat[4];
    }
};

// Run image detection.
// Example:
//      Detect detect("yolov8n_batch8.trt");
//      cv::Mat image = cv::imread("data/images/60.png");
//      detect.Infer(image);
class Detect
{
    private:
        int input_width;  // input width
        int input_height; // input height
        int image_width;  // output width
        int image_height; // output height
        cv::Mat M;
        cv::Mat IM;
        AffineMatrix affine_matrix; // affine transformation matrix
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
        std::vector<void *> host_ptrs;
        std::vector<void *> device_ptrs;

        void letterbox(const cv::Mat &image, cv::Mat &out); // make letterbox for the image
        void preprocess(std::vector<FrameInfo> &images);      // preprocess the image
        void postprocess(std::vector<FrameInfo> &images); // postprocess the image
        void makePipe(bool warmup);
        void copyFromMat(cv::Mat &nchw);
        void infer();

        void nonMaxSuppression(std::vector<FrameInfo> &images, int batch_size); // non-maximum suppression
        float iou(const cv::Rect rect1, const cv::Rect rect2);    // calculate the IOU of two rectangles

    public:
        Detect(std::string const &engine_path);                       // load the engine
        ~Detect();                                                    // unload the engine
        void detect(std::vector<FrameInfo> &images); // detect the image
        void engineInfo();                                            // print the engine information
};

class Segment
{
    private:
        int input_width;  // input width
        int input_height; // input height
        int image_width;  // output width
        int image_height; // output height
        cv::Mat M;
        cv::Mat IM;
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
        std::vector<void *> host_ptrs;
        std::vector<void *> device_ptrs;
        // cublas
        cublasHandle_t cublas_handle;
        cublasStatus_t cublas_status;

        void preprocess(std::vector<CropInfo> &crops);      // preprocess the meter crops
        void postprocess(std::vector<CropInfo> &crops); // postprocess the image
        void makePipe(bool warmup);
        void infer();

        void nonMaxSuppression(std::vector<CropInfo> &crops, int batch_size); // non-maximum suppression
        float iou(const cv::Rect rect1, const cv::Rect rect2);    // calculate the IOU of two rectangles
        void processMask(std::vector<CropInfo> &crops); // combine output0 and output1 to get masks

    public:
        Segment(std::string const &engine_path);                       // load the engine
        ~Segment();                                                    // unload the engine
        void segment(std::vector<CropInfo> &crops); // detect the image
        void engineInfo();                                            // print the engine information
};

#endif