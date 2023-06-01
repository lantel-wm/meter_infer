#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <algorithm>
#include <string>
#include <vector>
#include <fstream>
#include <time.h>

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
    cudaSetDevice(0);
    LOG(INFO) << "cuda device set to 0";

    // load the engine file
    this->engine_path = ENGINE_PATH + engine_filename;
    std::ifstream file(this->engine_path, std::ios::binary);
    LOG_ASSERT(file.good());   
    LOG(INFO) << "engine file opened: " << engine_filename;

    // read the enfine file into a buffer
    file.seekg(0, std::ios::end); // change the position of the stream to the end
    auto size = file.tellg();            // get the size of the engine file
    file.seekg(0, std::ios::beg); // change the position of the stream to the beginning
    LOG(INFO) << "engine file size: " << size;
    char *trtModelStream = new char[size];    // allocate a buffer to store the engine file
    LOG_ASSERT(trtModelStream) << "Failed to allocate buffer for engine file";
    file.read(trtModelStream, size);   // read the engine file into the buffer
    file.close();

    LOG(INFO) << "engine file loaded into buffer";

    // set image size to 1920x1080
    this->image_width = 1920;
    this->image_height = 1080;

    // set input size to 640x640
    this->input_width = IN_WIDTH;
    this->input_height = IN_HEIGHT;

    // load engine
    this->runtime = createInferRuntime(glogger);
    LOG_ASSERT(this->runtime != nullptr) << "Failed to create infer runtime";

    // deserialize the engine file
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    LOG_ASSERT(this->engine != nullptr) << "Failed to deserialize engine file " << engine_path;

    // create execution context
    this->context = this->engine->createExecutionContext();
    LOG_ASSERT(this->context != nullptr) << "Failed to create execution context";

    LOG_ASSERT(this->engine->getNbBindings() == 2) << "Invalid detection engine file: " << this->engine_path;

    LOG(INFO) << "Successfully loaded engine file " << engine_path;

    // // free the buffer
    delete[] trtModelStream;

    cudaStreamCreate(&this->stream);
    this->num_bindings = this->engine->getNbBindings();
    LOG(INFO) << "num_bindings: " << this->num_bindings;

    //  get binding info
    for (int i = 0; i < this->num_bindings; ++i)
	{
		Binding binding;
		Dims dims;
		DataType dtype = this->engine->getBindingDataType(i);
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
				OptProfileSelector::kMAX);
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

    // allocate memory for input and output 
    this->makePipe(true);
}

Detect::~Detect()
{
    this->context->destroy();
	this->engine->destroy();
	this->runtime->destroy();
	cudaStreamDestroy(this->stream);
	for (auto& ptr : this->device_ptrs)
	{
		CUDA_CHECK(cudaFree(ptr));
	}

	for (auto& ptr : this->host_ptrs)
	{
		CUDA_CHECK(cudaFreeHost(ptr));
	}
}

void Detect::engineInfo()
{
    LOG(INFO) << "--------------------------------------------------------------";
    LOG(INFO) << "ENGINE INFO:";
    LOG(INFO) << "num_bindings: " << this->num_bindings;
    LOG(INFO) << "num_inputs: " << this->num_inputs;
    LOG(INFO) << "num_outputs: " << this->num_outputs;
    for (int i = 0; i < this->num_inputs; ++i)
    {
        LOG(INFO) << "input" << i << " binding name: " << this->input_bindings[i].name;
        std::string input_dims = "[";
        for (int j = 0; j < this->input_bindings[i].dims.nbDims; ++j)
        {
           input_dims += std::to_string(this->input_bindings[i].dims.d[j]);
           if (j != this->input_bindings[i].dims.nbDims - 1)
                input_dims += ", ";
           else
                input_dims += "]";
        }
        LOG(INFO) << "input" << i << " binding dimension: " << input_dims;
        LOG(INFO) << "input" << i << " binding size: " << this->input_bindings[i].size;
        LOG(INFO) << "input" << i << " binding data type: " << this->input_bindings[i].dsize;
    }

    for (int i = 0; i < this->num_outputs; ++i)
    {
        LOG(INFO) << "output" << i << " binding name: " << this->output_bindings[i].name;
        std::string output_dims = "[";
        for (int j = 0; j < this->output_bindings[i].dims.nbDims; ++j)
        {
           output_dims += std::to_string(this->output_bindings[i].dims.d[j]);
           if (j != this->output_bindings[i].dims.nbDims - 1)
                output_dims += ", ";
           else
                output_dims += "]";
        }
        LOG(INFO) << "output" << i << " binding dimension: " << output_dims;
        LOG(INFO) << "output" << i << " binding size: " << this->output_bindings[i].size;
        LOG(INFO) << "output" << i << " binding data type: " << this->output_bindings[i].dsize;
    }
    LOG(INFO) << "--------------------------------------------------------------";
}

// preprocess the input image
void Detect::letterbox(const cv::Mat& image, cv::Mat& nchw)
{
    // set the affine transformation matrix
    LOG(INFO) << "making letterbox";
    LOG(INFO) << "image size: " << image.cols << "x" << image.rows;
    this->image_width = image.cols;
    this->image_height = image.rows;

    float scale = std::min(float(this->input_width) / image.cols, float(this->input_height) / image.rows);
    float delta_x = (-scale * image.cols + this->input_width) / 2;
    float delta_y = (-scale * image.rows + this->input_height) / 2;
    LOG(INFO) << "scale: " << scale << ", delta_x: " << delta_x << ", delta_y: " << delta_y;

    // M = [[scale, 0, delta_x], [0, scale, delta_y]]
    this->M = cv::Mat::zeros(2, 3, CV_32FC1);
    this->M.at<float>(0, 0) = scale;
    this->M.at<float>(1, 1) = scale;
    this->M.at<float>(0, 2) = delta_x;
    this->M.at<float>(1, 2) = delta_y;
    LOG(INFO) << "M: " << this->M;

    // apply the affine transformation (letterbox)
    auto t1 = clock();
    cv::Size size(this->input_width, this->input_height);
    cv::warpAffine(image, 
        nchw,
        this->M,
        size, 
        cv::INTER_LINEAR, 
        cv::BORDER_CONSTANT, 
        cv::Scalar(114, 114, 114)
    );
    // cv::invertAffineTransform(this->M, this->IM);
    auto t2 = clock();
    LOG(WARNING) << "warpAffine time: " << (t2 - t1) / 1000.0f << "ms";
    // cv::imwrite("letterbox.png", nchw);

    // blobFromImage:
    // 1. BGR to RGB
    // 2. /255.0, normalize to [0, 1]
    // 3. H,W,C to C,H,W
    t1 = clock();
    nchw = cv::dnn::blobFromImage(
        nchw, 
        1.0f / 255.0f, 
        nchw.size(), 
        cv::Scalar(0.0f, 0.0f, 0.0f), 
        true, 
        false, 
        CV_32F
    );
    t2 = clock();
    LOG(WARNING) << "blobFromImage time: " << (t2 - t1) / 1000.0f << "ms";

    LOG(INFO) << "input size after preprocess: "
              << "[" << nchw.size[0] << ", " << nchw.size[1]
              << ", " << nchw.size[2] << ", " << nchw.size[3] << "]";
}

float Detect::iou(const cv::Rect rect1, const cv::Rect rect2)
{
    cv::Rect intersection = rect1 & rect2;
    float intersection_area = intersection.area();
    float union_ = rect1.area() + rect2.area() - intersection_area;
    return intersection_area / union_;
}

void Detect::nonMaxSuppression(std::vector<frameInfo> &images, int batch_size)
{

    for (int l = 0; l < batch_size; l++)
    {
        // sort the results by confidence in descending order
        std::vector<DetObject> det_objs = images[l].det_objs;
        std::sort(det_objs.begin(), det_objs.end(), [](DetObject &a, DetObject &b) { return a.conf > b.conf; });

        DUMP_OBJ_INFO(det_objs);
        std::vector<bool> keep(det_objs.size(), true);
        for (int i = 0; i < det_objs.size(); i++)
        {
            if (keep[i])
            {
                for (int j = i + 1; j < det_objs.size(); j++)
                {
                    if (keep[j])
                    {
                        if (this->iou(det_objs[i].rect, det_objs[j].rect) > NMS_THRESH)
                        {
                            keep[j] = false;
                        }
                    }
                }
            }
        }

        std::vector<DetObject> det_objs_nms; 

        for (int i = 0; i < keep.size(); i++)
        {
            LOG(INFO) << "keep[" << i << "]: " << keep[i];
            if (keep[i])
            {
                det_objs_nms.push_back(det_objs[i]);
            }
        }

        DUMP_OBJ_INFO(det_objs_nms);

        images[l].det_objs = det_objs_nms;


        // for (int i = 0, j = 0; i < det_objs.size(); i++, j++)
        // {
        //     if (!keep[j])
        //     {
        //         det_objs.erase(det_objs.begin() + i);
        //         i--;
        //     }
        // }
    }
    LOG(INFO) << "non_max_suppresion done";
}

void Detect::postprocess(std::vector<frameInfo> &images)
{
    int batch_size = this->output_bindings[0].dims.d[0];
	int det_length = this->output_bindings[0].dims.d[1];
    int num_dets = this->output_bindings[0].dims.d[2];
    float* output = static_cast<float*>(this->host_ptrs[0]);
    LOG(INFO) << "batch_size: " << batch_size << ", det_length: " << det_length << ", num_dets: " << num_dets;

    // M = [[scale, 0, delta_x], [0, scale, delta_y]]
	float dw = this->affine_matrix.mat[2];
	float dh = this->affine_matrix.mat[5];
	float ratio = this->affine_matrix.inv_mat[0];
    LOG(INFO) << "dw: " << dw << ", dh: " << dh << ", ratio: " << ratio;

    for (int i = 0; i < batch_size; i++)
    {
        for (int k = 0; k < num_dets; k++)
        {
            // for (int j = 0; j < det_length; j++)
            // {
            //     LOG(INFO) << "output[" << i << "][" << j << "][" << k << "]: " << GET(output, i, j, k);
            // }
            int class_id = ARGMAX3(GET(output, i, 4, k), GET(output, i, 5, k), GET(output, i, 6, k));
            float conf = GET(output, i, 4 + class_id, k);
            if (conf < CONF_THRESH) continue;

            float x = GET(output, i, 0, k);
            float y = GET(output, i, 1, k);
            float w = GET(output, i, 2, k);
            float h = GET(output, i, 3, k);

            LOG(INFO) << "x: " << x << ", y: " << y << ", w: " << w << ", h: " << h;

            float x1 = (x - w / 2.f) - dw;
            float y1 = (y - h / 2.f) - dh;
            float x2 = (x + w / 2.f) - dw;
            float y2 = (y + h / 2.f) - dh;

            x1 = clamp(x1 * ratio, 0.f, this->image_width);
            y1 = clamp(y1 * ratio, 0.f, this->image_height);
            x2 = clamp(x2 * ratio, 0.f, this->image_width);
            y2 = clamp(y2 * ratio, 0.f, this->image_height);

            LOG(INFO) << "x1: " << x1 << ", y1: " << y1 << ", x2: " << x2 << ", y2: " << y2;

            DetObject det_obj;
            det_obj.rect = cv::Rect(x1, y1, x2 - x1, y2 - y1);
            det_obj.conf = conf;
            det_obj.batch_id = i;
            det_obj.class_id = class_id;
            det_obj.name = CLASS_NAMES[class_id];
            images[i].det_objs.push_back(det_obj);
        }
    }
    
    for (int i = 0; i < batch_size; i++)
    {
        LOG(INFO) << "detected objects in batch " << i << " before nms: " << images[i].det_objs.size();
    }
    
    this->nonMaxSuppression(images, batch_size);

    for (int i = 0; i < batch_size; i++)
    {
        LOG(INFO) << "detected objects in batch " << i << " after nms: " << images[i].det_objs.size();
    }
}

void Detect::makePipe(bool warmup)
{

	for (auto& bindings : this->input_bindings)
	{
		void* d_ptr; // device pointer
		CUDA_CHECK(cudaMallocAsync(
			&d_ptr,
			bindings.size * bindings.dsize,
			this->stream)
		);
		this->device_ptrs.push_back(d_ptr);
	}

	for (auto& bindings : this->output_bindings)
	{
		void* d_ptr, * h_ptr; // device pointer, host pointer
		size_t size = bindings.size * bindings.dsize;
        LOG(INFO) << "output size for cudaMalloc: " << size;
		CUDA_CHECK(cudaMallocAsync(
			&d_ptr,
			size,
			this->stream)
		);
		CUDA_CHECK(cudaHostAlloc(
			&h_ptr,
			size,
			0)
		);
		this->device_ptrs.push_back(d_ptr);
		this->host_ptrs.push_back(h_ptr);
	}

	if (warmup)
	{
		for (int i = 0; i < WARMUP_TIME; i++)
		{
			for (auto& bindings : this->input_bindings)
			{
				size_t size = bindings.size * bindings.dsize;
				void* h_ptr = malloc(size);
				memset(h_ptr, 0, size);
				CUDA_CHECK(cudaMemcpyAsync(
					this->device_ptrs[0],
					h_ptr,
					size,
					cudaMemcpyHostToDevice,
					this->stream)
				);
				free(h_ptr);
			}
			this->infer();
		}
		LOG(INFO) << "model warmup " << WARMUP_TIME  << " times";

	}
}

void Detect::infer()
{
    this->context->enqueueV2(
		this->device_ptrs.data(),
		this->stream,
		nullptr
	);
	for (int i = 0; i < this->num_outputs; i++)
	{
		size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
		CUDA_CHECK(cudaMemcpyAsync(this->host_ptrs[i],
			this->device_ptrs[i + this->num_inputs],
			osize,
			cudaMemcpyDeviceToHost,
			this->stream)
		);

	}
	cudaStreamSynchronize(this->stream);
}

void Detect::copyFromMat(cv::Mat& nchw)
{
	this->context->setBindingDimensions(
		0,
		Dims
			{
				4,
				{ 1, 3, this->input_height, this->input_width }
			}
	);
    LOG(INFO) << "binding dimensions set";

	CUDA_CHECK(cudaMemcpyAsync(
		this->device_ptrs[0],
		nchw.ptr<float>(),
		nchw.total() * nchw.elemSize(),
		cudaMemcpyHostToDevice,
		this->stream)
	);
}

// run detection on the image
void Detect::detect(std::vector<frameInfo> &images)
{
    // preprocess input
    cv::Mat nchw;
    auto t1 = clock();
    // this->letterbox(image, nchw);
    this->preprocess(images);
    auto t2 = clock();
    LOG(WARNING) << "image processed in " << (t2 - t1) / 1000.0 << " ms";

    // // copy to device
    // t1 = clock();
    // this->copyFromMat(nchw);
    // t2 = clock();
    // LOG(WARNING) << "image copied to device in " << (t2 - t1) / 1000.0 << " ms";

    // run inference
    t1 = clock();
    this->infer();
    t2 = clock();
    LOG(WARNING) << "inference done in " << (t2 - t1) / 1000.0 << " ms";

    // postprocess output
    t1 = clock();
    this->postprocess(images);
    t2 = clock();
    LOG(WARNING) << "postprocess done in " << (t2 - t1) / 1000.0 << " ms";
}