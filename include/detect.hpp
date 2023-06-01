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
    int batch_id;
    int class_id;
    float conf;
    float reading;
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
        0,        0,                     1
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
        void preprocess(std::vector<cv::Mat> &images);      // preprocess the image
        void postprocess(std::vector<std::vector<DetObject> >  &det_objs); // postprocess the image
        void makePipe(bool warmup);
        void copyFromMat(cv::Mat &nchw);
        void infer();

        void nonMaxSuppression(std::vector<std::vector<DetObject> >  &det_objs, int batch_size); // non-maximum suppression
        float iou(const cv::Rect rect1, const cv::Rect rect2);    // calculate the IOU of two rectangles

    public:
        Detect(std::string const &engine_path);                       // load the engine
        ~Detect();                                                    // unload the engine
        void detect(std::vector<cv::Mat> &images, std::vector<std::vector<DetObject> >  &results); // detect the image
        void engineInfo();                                            // print the engine information
};

#endif