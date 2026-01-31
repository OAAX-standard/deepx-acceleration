#include "runtime_core.h"

extern "C" {
#include "tensors_struct.h"
}

#include <memory>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <string.h>
#include <stdio.h>
#include <thread>
#include <fstream>

#include <dxrt/dxrt_api.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

struct JobData {
    int job_id;
    void *outputs_ptr; 
    tensors_struct *input_tensors;
    std::vector<std::shared_ptr<dxrt::Tensor>> dxrt_outputs;
};

static std::shared_ptr<spdlog::logger> logger;

static dxrt::InferenceEngine *inference_engine = nullptr;
static std::vector<uint64_t> OutputTensorSizes;

static size_t OUTPUTS_POOL_CAPACITY = 0;
static size_t NumDevice = 0;

static std::queue<void*> outputs_ptr_pool;
static std::mutex outputs_pool_mutex;
static std::condition_variable outputs_pool_cv;

static std::queue<JobData> job_data_queue;
static std::mutex job_data_queue_mutex;
static std::condition_variable job_data_queue_cv;

static std::queue<JobData> output_queue;
static std::mutex output_queue_mutex;
static std::condition_variable output_queue_cv;

static std::atomic<bool> stop_wait_thread{false};
static std::atomic<bool> wait_thread_started{false};
static std::thread wait_thread;

static tensors_struct *create_output_tensors_struct();
static tensors_struct *copy_dxrt_outputs_to_output_tensors_struct(const std::vector<std::shared_ptr<dxrt::Tensor>> &outputs, tensors_struct *output_tensors);
static tensor_data_type mapDataTypeToTensorDataType(dxrt::DataType dtype);
static void wait_loop();

static tensors_struct *create_output_tensors_struct() {
    int num_tensors = OutputTensorSizes.size();
    if (num_tensors == 0) {
        return nullptr;
    }
    
    tensors_struct *tensors = (tensors_struct *)malloc(sizeof(tensors_struct));
    if (tensors == nullptr) {
        return nullptr;
    }
    
    tensors->num_tensors = num_tensors;
    
    tensors->names = (char **)malloc(num_tensors * sizeof(char *));
    if (tensors->names == nullptr) {
        free(tensors);
        return nullptr;
    }
    
    tensors->data_types = (tensor_data_type *)malloc(num_tensors * sizeof(tensor_data_type));
    if (tensors->data_types == nullptr) {
        free(tensors->names);
        free(tensors);
        return nullptr;
    }
    
    tensors->ranks = (size_t *)malloc(num_tensors * sizeof(size_t));
    if (tensors->ranks == nullptr) {
        free(tensors->data_types);
        free(tensors->names);
        free(tensors);
        return nullptr;
    }
    
    tensors->shapes = (size_t **)malloc(num_tensors * sizeof(size_t *));
    if (tensors->shapes == nullptr) {
        free(tensors->ranks);
        free(tensors->data_types);
        free(tensors->names);
        free(tensors);
        return nullptr;
    }
    
    tensors->data = (void **)malloc(num_tensors * sizeof(void *));
    if (tensors->data == nullptr) {
        free(tensors->shapes);
        free(tensors->ranks);
        free(tensors->data_types);
        free(tensors->names);
        free(tensors);
        return nullptr;
    }
        
    for (size_t i = 0; i < num_tensors; i++) {
        tensors->names[i] = nullptr;
        tensors->data_types[i] = DATA_TYPE_UNDEFINED;
        tensors->ranks[i] = 0;
        tensors->shapes[i] = nullptr;
        tensors->data[i] = malloc(OutputTensorSizes[i]);
        if (tensors->data[i] == nullptr) {
            deep_free_tensors_struct(tensors);
            return nullptr;
        }
    }
    
    return tensors;
}

static tensors_struct *copy_dxrt_outputs_to_output_tensors_struct(const std::vector<std::shared_ptr<dxrt::Tensor>> &outputs, tensors_struct *output_tensors) {
    size_t num_output_tensors = outputs.size();
    if (output_tensors == nullptr) {
        return nullptr;
    }
    if (num_output_tensors == 0 || output_tensors->num_tensors != num_output_tensors) {
        spdlog::error("Output tensor size mismatch: dxrt_outputs={}, output_tensors_struct={}",
                      num_output_tensors, output_tensors->num_tensors);
        return nullptr;
    }
    
    for (size_t i = 0; i < num_output_tensors; i++) {
        const auto& output = outputs[i];
        
        auto name = output->name();
        auto shape = output->shape();
        auto dtype = output->type();
        auto data = output->data();
        
        output_tensors->names[i] = strdup(name.c_str());
        if (!output_tensors->names[i]) {
            spdlog::error("Failed to allocate name for tensor {}", i);
            deep_free_tensors_struct(output_tensors);
            return nullptr;
        }
        
        output_tensors->ranks[i] = shape.size();
        output_tensors->shapes[i] = (size_t *)malloc(shape.size() * sizeof(size_t));
        if (!output_tensors->shapes[i]) {
            spdlog::error("Failed to allocate shape for tensor {}", i);
            deep_free_tensors_struct(output_tensors);
            return nullptr;
        }
        
        for (size_t j = 0; j < shape.size(); j++) {
            output_tensors->shapes[i][j] = shape[j];
        }

        output_tensors->data_types[i] = mapDataTypeToTensorDataType(dtype);
        
        memcpy(output_tensors->data[i], data, OutputTensorSizes[i]);
    }
    return output_tensors;
}

static tensor_data_type mapDataTypeToTensorDataType(dxrt::DataType dtype) {
    switch (dtype) {
    case dxrt::UINT8:
        return DATA_TYPE_UINT8;
    case dxrt::UINT16:
        return DATA_TYPE_UINT16;
    case dxrt::UINT32:
        return DATA_TYPE_UINT32;
    case dxrt::UINT64:
        return DATA_TYPE_UINT64;
    case dxrt::INT8:
        return DATA_TYPE_INT8;
    case dxrt::INT16:
        return DATA_TYPE_INT16;
    case dxrt::INT32:
        return DATA_TYPE_INT32;
    case dxrt::INT64:
        return DATA_TYPE_INT64;
    case dxrt::FLOAT:
        return DATA_TYPE_FLOAT;
    default:
        return DATA_TYPE_UNDEFINED;
    }
}

int runtime_initialization() {
    try {
        logger = spdlog::basic_logger_mt(runtime_name(), "runtime.log");
        spdlog::set_default_logger(logger);
        spdlog::set_level(spdlog::level::info);
        spdlog::info("Initializing the runtime environment");
    } catch (const spdlog::spdlog_ex &ex) {
        printf("Warning: Failed to create logger: %s, continuing without file logging\n", ex.what());
    }

    return 0;
}

int runtime_initialization_with_args(int length, const char **keys, const void **values) {
    (void)values;
    
    int ret = runtime_initialization();
    if (ret != 0) {
        return ret;
    }

    spdlog::info("Runtime initialized with arguments");
    for (int i = 0; i < length; i++) {
        spdlog::debug("Using Key: {}", keys[i]);
    }

    return 0;
}

int runtime_model_loading(const char *file_path) {
    {
        std::ifstream f(file_path, std::ios::binary);
        if (!f) {
            spdlog::error("Model file does not exist: {}", file_path);
            return 1;
        }
    }

    spdlog::info("Loading model from: {}", file_path);

    try {
        inference_engine = new dxrt::InferenceEngine(std::string(file_path));
        if (inference_engine == nullptr) {
            spdlog::error("Failed to create inference engine");
            return 1;
        }

        NumDevice = dxrt::DeviceStatus::GetDeviceCount();
        OUTPUTS_POOL_CAPACITY = NumDevice * 10;

        OutputTensorSizes = inference_engine->GetOutputTensorSizes();
        uint64_t OutputSize = inference_engine->GetOutputSize();
        {
            std::lock_guard<std::mutex> lock(outputs_pool_mutex);
            while (!outputs_ptr_pool.empty()) outputs_ptr_pool.pop();
            for (size_t i = 0; i < OUTPUTS_POOL_CAPACITY; ++i) {
                void* outputs_ptr = malloc(OutputSize);
                if (!outputs_ptr) {
                    spdlog::error("Failed to allocate output buffer {}", i);
                    continue;
                }
                outputs_ptr_pool.push(outputs_ptr);
            }
            spdlog::info("Initialized outputs_ptr_pool with {} buffers for {} devices", outputs_ptr_pool.size(), NumDevice);
        }
        outputs_pool_cv.notify_one();

        stop_wait_thread.store(false);
        try {
            wait_thread = std::thread(wait_loop);
            wait_thread_started.store(true);
        } catch (...) {
            spdlog::error("Failed to create wait thread");
            {
                std::lock_guard<std::mutex> lock(outputs_pool_mutex);
                while (!outputs_ptr_pool.empty()) {
                    free(outputs_ptr_pool.front());
                    outputs_ptr_pool.pop();
                }
            }
            if (inference_engine) { delete inference_engine; inference_engine = nullptr; }
            return 1;
        }

        return 0;
    } catch (const std::exception& e) {
        spdlog::error("Failed to load model: {}", e.what());
        return 1;
    }
}

int send_input(tensors_struct *input_tensors) {

    if (input_tensors->num_tensors != 1) {
        spdlog::error("[send_input] Invalid number of input tensors: {}", input_tensors->num_tensors);
        return 1;
    }
    
    void *outputs_ptr = nullptr;
    {
        std::unique_lock<std::mutex> lock(outputs_pool_mutex);
        outputs_pool_cv.wait(lock, [](){ return !outputs_ptr_pool.empty(); });
        outputs_ptr = outputs_ptr_pool.front();
        outputs_ptr_pool.pop();
    }

    int job_id = -1;
    try {
        job_id = inference_engine->RunAsync(static_cast<uint8_t *>(input_tensors->data[0]), nullptr, outputs_ptr);
    } catch (const std::exception& e) {
        spdlog::error("[send_input] Failed to run inference : {}", e.what());
        {
            std::lock_guard<std::mutex> lock(outputs_pool_mutex);
            outputs_ptr_pool.push(outputs_ptr);
            outputs_pool_cv.notify_one();
        }
        if (input_tensors) deep_free_tensors_struct(input_tensors);
        return 1;
    }

    JobData job_data;
    job_data.job_id = job_id;
    job_data.input_tensors = input_tensors;
    job_data.outputs_ptr = outputs_ptr;

    {
        std::unique_lock<std::mutex> lock(job_data_queue_mutex);
        job_data_queue.push(job_data);
        job_data_queue_cv.notify_one();
    }

    return 0;
}

static void wait_loop() {
    while (true) {
        JobData job_data{};
        {
            std::unique_lock<std::mutex> lock(job_data_queue_mutex);
            job_data_queue_cv.wait(lock, [](){ return stop_wait_thread.load() || !job_data_queue.empty(); });
            if (stop_wait_thread.load() && job_data_queue.empty()) {
                break;
            }
            job_data = job_data_queue.front();
            job_data_queue.pop();
        }

        try {
            job_data.dxrt_outputs = inference_engine->Wait(job_data.job_id);
        } catch (...) {
            spdlog::error("[wait_loop] Failed to wait for outputs. job_id: {}", job_data.job_id);
            if (job_data.input_tensors) deep_free_tensors_struct(job_data.input_tensors);
            if (job_data.outputs_ptr) {
                std::lock_guard<std::mutex> lock(outputs_pool_mutex);
                outputs_ptr_pool.push(job_data.outputs_ptr);
                outputs_pool_cv.notify_one();
            }
            continue;
        }

        if (job_data.input_tensors) deep_free_tensors_struct(job_data.input_tensors);

        {
            std::lock_guard<std::mutex> lock(output_queue_mutex);
            output_queue.push(job_data);
            output_queue_cv.notify_one();
        }
    }
}

int receive_output(tensors_struct **output_tensors) {

    JobData job_data;
    {
        std::unique_lock<std::mutex> lock(output_queue_mutex);
        output_queue_cv.wait(lock, [](){ return stop_wait_thread.load() || !output_queue.empty(); });
        if (stop_wait_thread.load() && output_queue.empty()) {
            *output_tensors = nullptr;
            return 1;
        }
        job_data = output_queue.front();
        output_queue.pop();
    }

    tensors_struct *output_tensors_struct = create_output_tensors_struct();
    if (!output_tensors_struct) {
        spdlog::error("[receive_output] Failed to allocate output tensors");
        if (job_data.outputs_ptr) {
            std::lock_guard<std::mutex> lock(outputs_pool_mutex);
            outputs_ptr_pool.push(job_data.outputs_ptr);
            outputs_pool_cv.notify_one();
        }
        *output_tensors = nullptr;
        return 1;
    }

    *output_tensors = copy_dxrt_outputs_to_output_tensors_struct(job_data.dxrt_outputs, output_tensors_struct);
    if (*output_tensors == nullptr) {
        spdlog::error("[receive_output] Failed to convert dxrt outputs to output tensors");
        if (job_data.outputs_ptr) {
            std::lock_guard<std::mutex> lock(outputs_pool_mutex);
            outputs_ptr_pool.push(job_data.outputs_ptr);
            outputs_pool_cv.notify_one();
        }
        return 1;
    }

    if (job_data.outputs_ptr) {
        std::lock_guard<std::mutex> lock(outputs_pool_mutex);
        outputs_ptr_pool.push(job_data.outputs_ptr);
        outputs_pool_cv.notify_one();
    }

    return 0;
}

int runtime_destruction() {
    spdlog::info("Destroying the runtime environment");

    stop_wait_thread.store(true);
    job_data_queue_cv.notify_all();
    output_queue_cv.notify_all();
    outputs_pool_cv.notify_all();

    {
        std::lock_guard<std::mutex> lock(output_queue_mutex);
        while (!output_queue.empty()) {
            JobData r = std::move(output_queue.front());
            output_queue.pop();
            if (r.input_tensors) deep_free_tensors_struct(r.input_tensors);
            if (r.outputs_ptr) free(r.outputs_ptr);
        }
    }

    {
        std::lock_guard<std::mutex> lock(job_data_queue_mutex);
        while (!job_data_queue.empty()) {
            JobData j = job_data_queue.front();
            job_data_queue.pop();
            if (j.input_tensors) deep_free_tensors_struct(j.input_tensors);
            if (j.outputs_ptr) free(j.outputs_ptr);
        }
    }

    {
        std::lock_guard<std::mutex> lock(outputs_pool_mutex);
        while (!outputs_ptr_pool.empty()) {
            void *outputs_ptr = outputs_ptr_pool.front();
            outputs_ptr_pool.pop();
            free(outputs_ptr);
        }
    }

    if (wait_thread_started.load()) {
        if (wait_thread.joinable()) {
            wait_thread.join();
        }
        wait_thread_started.store(false);
    }

    if (inference_engine != nullptr) {
        delete inference_engine;
        inference_engine = nullptr;
        spdlog::info("Inference engine destroyed");
    }

    spdlog::info("Runtime destruction completed");
    if (logger) {
        logger->flush();
        logger.reset();
    }

    return 0;
}

const char *runtime_error_message() { 
    return ""; 
}

const char *runtime_version() { 
    return OAAX_RUNTIME_VERSION; 
}

const char *runtime_name() { 
    return "DEEPX"; 
}
