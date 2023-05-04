#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <algorithm>
#include <string>
#include <vector>
#include <fstream>

#include "glog/logging.h"
#include "config.hpp"
#include "common.hpp"
#include "detect.hpp"

using namespace nvinfer1;

class Logger : public ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            // std::cout << msg << std::endl;
            LOG(WARNING) << msg;
    }
} glogger;

// Constructor for the Detect class.
// Sets the input size to 640x640.
// Loads the engine from the engine file.
Detect::Detect(std::string const &engine_filename)
{
    std::ifstream engine_file(engine_path, std::ios::binary);
    LOG_IF(FATAL, !engine_file.good());   

    // read the enfine file into a buffer
    size_t size = 0;
    engine_file.seekg(0, engine_file.end); // change the position of the stream to the end
    size = engine_file.tellg();            // get the size of the engine file
    engine_file.seekg(0, engine_file.beg); // change the position of the stream to the beginning
    char *trtModelStream = new char[size];    // allocate a buffer to store the engine file
    LOG_ASSERT(trtModelStream) << "Failed to allocate buffer for engine file";
    engine_file.read(trtModelStream, size);   // read the engine file into the buffer
    engine_file.close();

    // set input size to 640x640
    this->width = IN_WIDTH;
    this->height = IN_HEIGHT;

    // load engine
    this->engine_path = ENGINE_PATH + engine_filename;
    this->runtime = createInferRuntime(glogger);
    LOG_ASSERT(this->runtime != nullptr) << "Failed to create infer runtime";

    // deserialize the engine file
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    LOG_ASSERT(this->engine != nullptr) << "Failed to deserialize engine file " << engine_path;

    // create execution context
    this->context = this->engine->createExecutionContext();
    LOG_ASSERT(this->context != nullptr) << "Failed to create execution context";

    LOG_ASSERT(this->engine->getNbBindings() == 2) << "Invalid detection engine file: " << this->engine_path;

    LOG(INFO) << "Successfully loaded engine file " << engine_path;

    // // free the buffer
    // delete[] trtModelStream;

    cudaStreamCreate(&this->stream);
    this->num_bindings = this->engine->getNbBindings();

    for (int i = 0; i < this->num_bindings; ++i)
	{
		Binding binding;
		nvinfer1::Dims dims;
		nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
		std::string name = this->engine->getBindingName(i);
		binding.name = name;
		binding.dsize = type_to_size(dtype);

		bool IsInput = engine->bindingIsInput(i);
		if (IsInput)
		{
			this->num_inputs += 1;
			dims = this->engine->getProfileDimensions(
				i,
				0,
				nvinfer1::OptProfileSelector::kMAX);
			binding.size = get_size_by_dims(dims);
			binding.dims = dims;
			this->input_bindings.push_back(binding);
			// set max opt shape
			this->context->setBindingDimensions(i, dims);

		}
		else
		{
			dims = this->context->getBindingDimensions(i);
			binding.size = get_size_by_dims(dims);
			binding.dims = dims;
			this->output_bindings.push_back(binding);
			this->num_outputs += 1;
		}
	}
}

Detect::~Detect()
{
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
}

cv::Mat Detect::processInput(cv::Mat &image)
{
    // set the affine transformation matrix
    float scale = std::min(float(this->width) / image.cols, float(this->height) / image.rows);
    float Ox = (-scale * image.cols + this->width) / 2;
    float Oy = (-scale * image.rows + this->height) / 2;
    LOG(INFO) << "scale: " << scale << ", Ox: " << Ox << ", Oy: " << Oy;
    this->M = cv::Mat::zeros(2, 3, CV_32FC1);
    this->M.at<float>(0, 0) = scale;
    this->M.at<float>(1, 1) = scale;
    this->M.at<float>(0, 2) = Ox;
    this->M.at<float>(1, 2) = Oy;
    LOG(INFO) << "M: " << this->M;

    // apply the affine transformation (letterbox)
    cv::Mat image_processed;
    cv::warpAffine(image, image_processed, this->M, cv::Size(this->width, this->height), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    cv::invertAffineTransform(this->M, this->IM);

    // cv::imshow("image_affine", image_processed);
    // cv::waitKey(0);

    // blobFromImage:
    // 1. BGR to RGB
    // 2. /255.0, normalize to [0, 1]
    // 3. H,W,C to C,H,W
    image_processed = cv::dnn::blobFromImage(image_processed, 1.0f / 255.0f, image_processed.size(), cv::Scalar(0.0f, 0.0f, 0.0f), true, false, CV_32F);

    LOG(INFO) << "input size after preprocess: "
              << "[" << image_processed.size[0] << ", " << image_processed.size[1]
              << ", " << image_processed.size[2] << ", " << image_processed.size[3] << "]";
    return image_processed;
}

float Detect::iou(cv::Rect &rect1, cv::Rect &rect2)
{
    int x1 = std::max(rect1.x, rect2.x);
    int y1 = std::max(rect1.y, rect2.y);
    int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

    int intersection = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int union_ = rect1.width * rect1.height + rect2.width * rect2.height - intersection;

    return float(intersection) / union_;
}

void Detect::nonMaxSupression(std::vector<detectResult> &results)
{
    // sort the results by confidence in descending order
    std::sort(results.begin(), results.end(), [](detectResult &a, detectResult &b) { return a.conf > b.conf; });

    std::vector<bool> keep(results.size(), true);
    for (int i = 0; i < results.size(); i++)
    {
        if (keep[i])
        {
            for (int j = i + 1; j < results.size(); j++)
            {
                if (keep[j])
                {
                    if (this->iou(results[i].rect, results[j].rect) > NMS_THRESH)
                    {
                        keep[j] = false;
                    }
                }
            }
        }
    }

    for (int i = 0; i < results.size(); i++)
    {
        if (!keep[i])
        {
            results.erase(results.begin() + i);
            i--;
        }
    }
    LOG(INFO) << "postprocess (nms) done";
}

void Detect::processOutput(float *output, std::vector<detectResult> &results)
{
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        for (int k = 0; k < DET_OUT_CHANNEL1; k++)
        {
            int index[DET_OUT_CHANNEL0];
            for (int j = 0; j < DET_OUT_CHANNEL0; j++)
            {
                index[j] = k + DET_OUT_CHANNEL1 * (j + DET_OUT_CHANNEL0 * i);
            }

            float conf = output[index[4]];

            if (conf > CONF_THRESH)
            {
                detectResult result;
                float cx = output[index[0]];
                float cy = output[index[1]];
                float w = output[index[2]];
                float h = output[index[3]];
                result.rect = cv::Rect2i(int(cx - w / 2.0), int(cy - h / 2.0), int(w), int(h));
                result.conf = conf;
                result.class_id = int(output[index[5]]);
                results.push_back(result);
            }
        }
    }

    LOG(INFO) << results.size() << " results before nms";
    this->nonMaxSupression(results);
}

void Detect::Infer(cv::Mat &image, std::vector<detectResult> &results)
{
    // preprocess input
    cv::Mat image_processed = this->processInput(image);

    const int inputIndex = this->engine->getBindingIndex(INPUT_NAME);
    const int outputIndex = this->engine->getBindingIndex(OUTPUT_NAME0);
    LOG(INFO) << "inputIndex: " << inputIndex << ", outputIndex: " << outputIndex;

    void *buffers[2];
    // cudaSuccess = 0
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * IN_CHANNEL * IN_WIDTH * IN_HEIGHT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * DET_OUT_CHANNEL0 * DET_OUT_CHANNEL1 * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // copy input to device
    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], image_processed.data, BATCH_SIZE * IN_CHANNEL * IN_WIDTH * IN_HEIGHT * sizeof(float), cudaMemcpyHostToDevice, stream));

    // run inference
    this->context->enqueueV2(buffers, stream, nullptr);

    // copy output to host
    float *output = new float[BATCH_SIZE * DET_OUT_CHANNEL0 * DET_OUT_CHANNEL1];
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[outputIndex], BATCH_SIZE * DET_OUT_CHANNEL0 * DET_OUT_CHANNEL1 * sizeof(float), cudaMemcpyDeviceToHost, stream));

    // wait for inference to finish
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // release the stream and the buffers
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));

    // postprocess output
    this->processOutput(output, results);
}