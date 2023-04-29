#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>

#include "NvInfer.h"
#include "glog/logging.h"
#include "config.hpp"
#include "detect.hpp"

using namespace nvinfer1;

class Logger : public ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            // std::cout << msg << std::endl;
            LOG(WARNING) << msg;
    }
} logger;

Detect::Detect(std::string const &engine_filename)
{
    this->engine_path = config::ENGINE_PATH + engine_filename;
    this->runtime = createInferRuntime(logger);

    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file.good())
    {
        LOG(ERROR) << "Failed to open engine file " << engine_path;
        return;
    }

    size_t size = 0;
	engine_file.seekg(0, engine_file.end); 
	size = engine_file.tellg();	
	engine_file.seekg(0, engine_file.beg);
	char *modelStream = new char[size];
	engine_file.read(modelStream, size);
	engine_file.close();

    this->engine = this->runtime->deserializeCudaEngine(modelStream, size, nullptr);
    if (!this->engine)
    {
        LOG(ERROR) << "Failed to deserialize engine file " << engine_path;
        return;
    }

    this->context = this->engine->createExecutionContext();
    if (!this->context)
    {
        LOG(ERROR) << "Failed to create execution context";
        return;
    }

    delete[] modelStream;
}

Detect::~Detect()
{
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
}

void Detect::preprocess(cv::Mat &image)
{
}

void Detect::postprocess(cv::Mat &image)
{
}

void Detect::infer(cv::Mat &image)
{
}