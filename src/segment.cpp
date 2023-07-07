#include <algorithm>
#include <string>
#include <vector>
#include <fstream>
#include <time.h>

#include "glog/logging.h"
#include "config.hpp"
#include "common.hpp"
#include "yolo.hpp"

Logger glogger_seg;

// Constructor for the Segment class.
// Sets the input size to 640x640.
// Loads the engine from the engine file.
Segment::Segment(std::string const &engine_filename)
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
    this->runtime = createInferRuntime(glogger_seg);
    LOG_ASSERT(this->runtime != nullptr) << "Failed to create infer runtime";

    // deserialize the engine file
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    LOG_ASSERT(this->engine != nullptr) << "Failed to deserialize engine file " << engine_path;

    // create execution context
    this->context = this->engine->createExecutionContext();
    LOG_ASSERT(this->context != nullptr) << "Failed to create execution context";

    LOG_ASSERT(this->engine->getNbBindings() == 3) << "Invalid segmention engine file: " << this->engine_path;

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

    //cublas init
    cublas_status = cublasCreate(&cublas_handle);
    LOG_ASSERT(cublas_status == CUBLAS_STATUS_SUCCESS) << "CUBLAS initialization failed!\n";

}

Segment::~Segment()
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
    cublasDestroy(cublas_handle);
}

void Segment::engineInfo()
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

float Segment::iou(const cv::Rect rect1, const cv::Rect rect2)
{
    cv::Rect intersection = rect1 & rect2;
    float intersection_area = intersection.area();
    float union_ = rect1.area() + rect2.area() - intersection_area;
    return intersection_area / union_;
}

void Segment::nonMaxSuppression(std::vector<CropInfo> &crops, int batch_size)
{

    for (int l = 0; l < batch_size; l++)
    {
        // sort the results by confidence in descending order
        std::vector<DetObject> det_objs = crops[l].det_objs;
        std::vector<DetObject> det_objs_nms; 
        std::sort(det_objs.begin(), det_objs.end(), [](DetObject &a, DetObject &b) { return a.conf > b.conf; });

        // DUMP_OBJ_INFO(det_objs);

        while (det_objs.size() > 0)
        {
            DetObject det_obj = det_objs[0];
            det_objs.erase(det_objs.begin());
            det_objs_nms.push_back(det_obj);

            for (int i = 0; i < det_objs.size(); i++)
            {
                if (this->iou(det_obj.rect, det_objs[i].rect) > NMS_THRESH)
                {
                    det_objs.erase(det_objs.begin() + i);
                    i--;
                }
            }
        }
        // DUMP_OBJ_INFO(det_objs_nms);

        crops[l].det_objs = det_objs_nms;
    }
    LOG(INFO) << "non_max_suppresion done";
}

void Segment::postprocess(std::vector<CropInfo> &crops)
{
    int batch_size, det_length, num_dets;
    bool flag = false;
    float t1, t2;

    batch_size = crops.size();
    for (auto &ob: this->output_bindings)
    {
        if (ob.name == "output0" && ob.dims.nbDims == 3)
        {
            det_length = ob.dims.d[1];
            num_dets = ob.dims.d[2];
            flag = true;
        }
    }
    
    LOG_ASSERT(flag) << " output binding dims is not 3";

    float* output0 = static_cast<float*>(this->host_ptrs[1]);
    LOG(INFO) << "batch_size: " << batch_size << ", det_length: " << det_length << ", num_dets: " << num_dets;

    
    for (int i = 0; i < batch_size; i++)
    {
        for (int k = 0; k < num_dets; k++)
        {
            int class_id = ARGMAX2(
                GET(output0, i, 4, k, batch_size, det_length, num_dets), 
                GET(output0, i, 5, k, batch_size, det_length, num_dets)
                );
            float conf = GET(output0, i, 4 + class_id, k, batch_size, det_length, num_dets);
            // LOG(INFO) << "i: " << i << ", k: " << k << ", class_id: " << class_id << ", conf: " << conf;
            if (conf < CONF_THRESH) continue;

            float x = GET(output0, i, 0, k, batch_size, det_length, num_dets);
            float y = GET(output0, i, 1, k, batch_size, det_length, num_dets);
            float w = GET(output0, i, 2, k, batch_size, det_length, num_dets);
            float h = GET(output0, i, 3, k, batch_size, det_length, num_dets);

            // LOG(INFO) << "x: " << x << ", y: " << y << ", w: " << w << ", h: " << h;

            float x1 = x - w / 2.f;
            float y1 = y - h / 2.f;
            float x2 = x + w / 2.f;
            float y2 = y + h / 2.f;

            // LOG(INFO) << "x1: " << x1 << ", y1: " << y1 << ", x2: " << x2 << ", y2: " << y2;

            DetObject det_obj;
            det_obj.rect = cv::Rect(x1, y1, x2 - x1, y2 - y1);
            det_obj.conf = conf;
            det_obj.batch_id = i;
            det_obj.class_id = class_id;
            det_obj.name = CLASS_NAMES2[class_id];
            det_obj.mask_in = new float[32];
            for (int j = 0; j < 32; j++)
            {
                det_obj.mask_in[j] = GET(output0, i, j + 4 + 2, k, batch_size, det_length, num_dets);
            }
            crops[i].det_objs.push_back(det_obj);
        }
    }
    

    for (int i = 0; i < batch_size; i++)
    {
        LOG(INFO) << "detected objects in batch " << i << " before nms: " << crops[i].det_objs.size();
    }

    // t1 = clock();
    this->nonMaxSuppression(crops, batch_size);
    // t2 = clock();
    // LOG(WARNING) << "nms time: " << 1000 * (t2 - t1) * 1.0 / CLOCKS_PER_SEC << "ms";

    for (int i = 0; i < batch_size; i++)
    {
        LOG(INFO) << "detected objects in batch " << i << " after nms: " << crops[i].det_objs.size();
    }

    // t1 = clock();
    processMask(crops);
    // t2 = clock();
    // LOG(WARNING) << "process mask time: " << 1000 * (t2 - t1) * 1.0 / CLOCKS_PER_SEC << "ms";
}

void Segment::makePipe(bool warmup)
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

void Segment::infer()
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

// run segmention on the image
void Segment::segment(std::vector<CropInfo> &crops)
{
    // preprocess input
    // auto t1 = clock();
    // this->letterbox(image, nchw);
    this->preprocess(crops);
    // auto t2 = clock();
    // LOG(WARNING) << "image processed in " << (t2 - t1) / 1000.0 << " ms";

    // // copy to device
    // t1 = clock();
    // this->copyFromMat(nchw);
    // t2 = clock();
    // LOG(WARNING) << "image copied to device in " << (t2 - t1) / 1000.0 << " ms";

    // run inference
    // t1 = clock();
    this->infer();
    // t2 = clock();
    // LOG(WARNING) << "inference done in " << (t2 - t1) / 1000.0 << " ms";

    // postprocess output
    // t1 = clock();
    this->postprocess(crops);
    // t2 = clock();
    // LOG(WARNING) << "postprocess done in " << (t2 - t1) / 1000.0 << " ms";
}